import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertModel, BertTokenizer
from annoy import AnnoyIndex
from tqdm import tqdm  # Import tqdm for progress bars
import copy
import os
import matplotlib.pyplot as plt
class ContrastiveLearningEncoder(nn.Module):
    """
    Encoder for contrastive learning to create high-quality text vectors
    """

    def __init__(self, pretrained_model="MiniRBT-h256-pt", vector_size=256):
        super(ContrastiveLearningEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.vector_size = vector_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get the [CLS] token representation
        cls_vector=outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_vector, p=2, dim=1)

    def encode_text(self, text, tokenizer, device):
        """Encode a single text into a vector"""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            vector = self.forward(inputs["input_ids"], inputs["attention_mask"])

        return vector.cpu().numpy()[0]


class AttentionMechanism(nn.Module):
    """
    Attention mechanism to focus on relevant neighboring instances
    """

    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        # self.attention = nn.Linear(input_dim, 1)

    def forward(self, query_vector, neighbor_vectors):
        """
        Apply attention to neighbor vectors based on their similarity to query vector

        Args:
            query_vector: Tensor of shape (batch_size, vector_dim)
            neighbor_vectors: Tensor of shape (batch_size, k_neighbors, vector_dim)

        Returns:
            attention_weighted_vector: Weighted sum of neighbor vectors
            attention_weights: The weights assigned to each neighbor
        """
        # Calculate similarity scores between query and neighbors
        batch_size, k_neighbors, vector_dim = neighbor_vectors.size()

        # Expand query vector to match neighbor vectors shape
        query_expanded = query_vector.unsqueeze(1).expand(-1, k_neighbors, -1)

        # # Calculate Euclidean distance (negative for higher values = closer)
        # distances = -torch.sqrt(torch.sum((query_expanded - neighbor_vectors) ** 2, dim=2))
        #
        # # Apply softmax to get attention weights
        # attention_weights = F.softmax(distances, dim=1)  # Shape: (batch_size, k_neighbors)
        # 将欧式距离改为余弦相似度
        similarity = F.cosine_similarity(query_expanded, neighbor_vectors, dim=2)
        attention_weights = F.softmax(similarity, dim=1)  # 直接使用相似度

        # Apply attention weights to neighbor vectors
        weighted_vectors = neighbor_vectors * attention_weights.unsqueeze(2)
        attention_weighted_vector = torch.sum(weighted_vectors, dim=1)  # Shape: (batch_size, vector_dim)

        return attention_weighted_vector, attention_weights


class BiLSTM(nn.Module):
    """
    BiLSTM model for text classification - replacing the CNN
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # The output dimension is twice the hidden dim because of bidirectionality
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sentence length]
        embedded = self.embedding(text)  # [batch size, sentence length, embedding dim]

        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1))

        return self.fc(hidden)


class LSTMCL_AM(nn.Module):
    """
    BiLSTM model with Contrastive Learning and Attention Mechanism
    """

    def __init__(self, bilstm, attention_mechanism, num_classes, embedding_dim, device):
        super(LSTMCL_AM, self).__init__()
        self.bilstm = bilstm
        self.attention_mechanism = attention_mechanism
        self.device = device

        # Final classification layer with concatenated features
        self.classifier = nn.Linear(embedding_dim * 2 + num_classes,
                                    num_classes)  # Original + text attention + label attention

    def forward(self, text, neighbor_text_vectors, neighbor_labels):
        """
        Forward pass with attention mechanism

        Args:
            text: Input text tensor
            neighbor_text_vectors: Tensor of shape (batch_size, k_neighbors, vector_dim)
            neighbor_labels: Tensor of shape (batch_size, k_neighbors, num_classes) - one-hot encoded
        """
        # Get original text features
        original_features = self.bilstm(text)

        # Apply attention to neighbor text vectors
        text_attention_vec, text_weights = self.attention_mechanism(original_features, neighbor_text_vectors)

        # Apply same attention weights to labels
        batch_size, k_neighbors = neighbor_labels.size(0), neighbor_labels.size(1)
        label_weights = text_weights.view(batch_size, k_neighbors, 1)
        weighted_labels = neighbor_labels * label_weights
        label_attention_vec = torch.sum(weighted_labels, dim=1)

        # Concatenate original features with attention-weighted vectors
        combined_features = torch.cat([original_features, text_attention_vec, label_attention_vec], dim=1)

        # Final classification
        output = self.classifier(combined_features)
        return output


class TextClassifier:
    def __init__(self, num_classes,
                 contrastive_epochs=20,
                 classifier_epochs=10,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 pretrained_model="MiniRBT-h256-pt",
                 k_neighbors=10):
        # 基础配置
        self.device = device
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        self.contrastive_epochs = contrastive_epochs
        self.classifier_epochs = classifier_epochs

        self.best_f1 = 0.0
        self.best_model_state = None
        self.best_epoch = 0

        # 初始化编码器和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.cl_encoder = ContrastiveLearningEncoder(pretrained_model).to(device)

        # 近邻检索系统初始化
        self.vector_size = 256
        self.annoy_index = None
        self.training_vectors = []
        self.training_labels = []

        # BiLSTM模型初始化
        vocab_size = self.tokenizer.vocab_size
        embedding_dim = 256
        hidden_dim = 128
        n_layers = 2
        dropout = 0.5
        pad_idx = self.tokenizer.pad_token_id

        self.bilstm = BiLSTM(
            vocab_size, embedding_dim, hidden_dim,
            embedding_dim, n_layers, dropout, pad_idx
        ).to(device)

        # 注意力机制和融合模型
        self.attention_mechanism = AttentionMechanism(embedding_dim).to(device)
        self.model = LSTMCL_AM(
            self.bilstm, self.attention_mechanism,
            num_classes, embedding_dim, device
        ).to(device)

        # ================= 优化器参数组分离 =================
        # 对比学习阶段参数组（BERT + BiLSTM）
        contrastive_params = [
            # BERT参数组（小学习率，强正则化）
            {
                'params': self.cl_encoder.parameters(),
                'lr': 1e-4,  # 初始学习率
                'weight_decay': 0.01,  # 权重衰减
                'betas': (0.9, 0.999)  # Adam参数
            },
            # BiLSTM参数组（中等学习率，适度正则化）
            {
                'params': self.bilstm.parameters(),
                'lr': 4e-4,
                'weight_decay': 0.001,
                'betas': (0.9, 0.999)
            }
        ]

        # 分类阶段参数组（注意力机制 + 分类层）
        classifier_params = [

            # 最终分类层参数
            {
                'params': self.model.classifier.parameters(),
                'lr': 1e-3,
                'weight_decay': 0.001,
                'betas': (0.9, 0.98)
            }
        ]

        # 初始化各阶段优化器
        self.contrastive_optimizer = torch.optim.AdamW(contrastive_params)
        self.classifier_optimizer = torch.optim.Adam(classifier_params)

        # 损失函数
        self.contrastive_loss_fn = self.supervised_contrastive_loss
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # 打印参数组信息（调试用）
        self._print_optimizer_groups()

    def _print_optimizer_groups(self):
        """打印优化器参数组配置"""
        print("\n=== Contrastive Learning 参数组 ===")
        for i, group in enumerate(self.contrastive_optimizer.param_groups):
            params_count = sum(p.numel() for p in group['params'])
            print(f"Group {i + 1}:")
            print(f"  Learning rate: {group['lr']:.1e}")
            print(f"  Weight decay: {group['weight_decay']}")
            print(f"  Betas: {group['betas']}")
            print(f"  参数数量: {params_count:,}")

        print("\n=== Classifier 参数组 ===")
        for i, group in enumerate(self.classifier_optimizer.param_groups):
            params_count = sum(p.numel() for p in group['params'])
            print(f"Group {i + 1}:")
            print(f"  Learning rate: {group['lr']:.1e}")
            print(f"  Weight decay: {group['weight_decay']}")
            print(f"  Betas: {group['betas']}")
            print(f"  参数数量: {params_count:,}")

    def save_contrastive_model(self, save_path="bilstm-am-contrastive_model.pt"):
        """
        保存对比学习模型及索引数据
        """
        # 创建保存目录(如果不存在)
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存模型参数
        state_dict = {
            'cl_encoder': self.cl_encoder.state_dict(),
            'bilstm': self.bilstm.state_dict(),
            'vector_size': self.vector_size,
            'num_classes': self.num_classes,
        }

        # 保存ANN索引和训练向量/标签
        if self.annoy_index and self.training_vectors and self.training_labels:
            # 保存Annoy索引
            index_path = f"{os.path.splitext(save_path)[0]}_index.ann"
            self.annoy_index.save(index_path)

            # 保存训练向量和标签
            vectors_labels = {
                'training_vectors': np.array(self.training_vectors),
                'training_labels': np.array(self.training_labels)
            }
            np.savez(f"{os.path.splitext(save_path)[0]}_vectors_labels.npz", **vectors_labels)

            # 将索引路径添加到状态字典
            state_dict['index_path'] = index_path
            state_dict['vectors_labels_path'] = f"{os.path.splitext(save_path)[0]}_vectors_labels.npz"

        # 保存模型状态
        torch.save(state_dict, save_path)
        print(f"对比学习模型保存到: {save_path}")

    def load_contrastive_model(self, load_path="bilstm-am-contrastive_model.pt"):
        """
        加载对比学习模型及索引数据
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        # 加载模型状态
        print(f"正在加载对比学习模型: {load_path}")
        state_dict = torch.load(load_path, map_location=self.device)

        # 加载编码器和BiLSTM参数
        self.cl_encoder.load_state_dict(state_dict['cl_encoder'])
        self.bilstm.load_state_dict(state_dict['bilstm'])

        # 验证向量大小和类别数
        assert state_dict['vector_size'] == self.vector_size, "向量维度不匹配"
        assert state_dict['num_classes'] == self.num_classes, "类别数不匹配"

        # 加载索引和向量/标签(如果存在)
        if 'index_path' in state_dict and 'vectors_labels_path' in state_dict:
            index_path = state_dict['index_path']
            vectors_labels_path = state_dict['vectors_labels_path']

            if os.path.exists(index_path) and os.path.exists(vectors_labels_path):
                # 加载ANN索引
                self.annoy_index = AnnoyIndex(self.vector_size, 'angular')
                self.annoy_index.load(index_path)

                # 加载训练向量和标签
                data = np.load(vectors_labels_path)
                self.training_vectors = data['training_vectors'].tolist()
                self.training_labels = data['training_labels'].tolist()

                print(f"成功加载近邻索引和{len(self.training_vectors)}个训练样本向量")
                return True
            else:
                print(f"警告: 索引文件或向量标签文件不存在")
                return False
        else:
            print(f"警告: 状态字典中没有索引路径信息")
            return False
    def _evaluate_validation(self, val_texts, val_labels, batch_size=64):
        """验证集评估函数"""
        # Set model to evaluation mode - FIX 1: Explicitly set all models to eval mode
        self.model.eval()
        self.bilstm.eval()
        self.cl_encoder.eval()
        self.attention_mechanism.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        # 修改 _evaluate_validation 函数中的标签转换代码
        val_labels = torch.tensor(val_labels, dtype=torch.long, device=self.device)
        # 创建验证数据集
        val_encodings = self.tokenizer(list(val_texts),
                                       padding=True, truncation=True,
                                       return_tensors="pt", max_length=512)

        # 获取邻居向量和标签
        neighbor_vectors, neighbor_labels = self._get_neighbors_for_batch(val_texts)

        val_dataset = torch.utils.data.TensorDataset(
            val_encodings['input_ids'].to(self.device),
            val_encodings['attention_mask'].to(self.device),
            torch.tensor(val_labels, device=self.device),
            neighbor_vectors,
            neighbor_labels
        )

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels, nn_vecs, nn_labs = batch

                outputs = self.model(input_ids, nn_vecs, nn_labs)
                loss = self.ce_loss_fn(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, f1

    def _get_neighbors_for_batch(self, texts):
        """批量获取邻居的向量和标签"""
        all_nn_vectors = []
        all_nn_labels = []

        for text in texts:
            nn_vectors, nn_labels = self._retrieve_neighbors(text)
            # 转换one-hot标签
            one_hot = np.zeros((self.k_neighbors, self.num_classes))
            for i, label in enumerate(nn_labels):
                one_hot[i, label] = 1
            all_nn_vectors.append(nn_vectors)
            all_nn_labels.append(one_hot)

        return (torch.tensor(np.array(all_nn_vectors), dtype=torch.float32, device=self.device),
                torch.tensor(np.array(all_nn_labels), dtype=torch.float32, device=self.device))

    def supervised_contrastive_loss(self, features, labels, temperature=0.07):
        """
        Supervised contrastive loss function as described in the paper
        """
        # 添加L2归一化（新增代码）
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)

        # Calculate distances between all pairs (using dot product for similarity)
        similarity_matrix = torch.matmul(features, features.T)

        # Mask for positive pairs (same class)
        mask = torch.eq(labels, labels.T).float()

        # Remove self-contrast cases
        mask = mask - torch.eye(batch_size, device=self.device)

        # Calculate positive pairs for each anchor
        pos_per_sample = mask.sum(dim=1)

        # Handle case with no positive pairs for a sample
        pos_per_sample = torch.clamp(pos_per_sample, min=1)

        # Apply temperature scaling
        similarity_matrix = similarity_matrix / temperature

        # For numerical stability
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - sim_max.detach()

        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix)

        # Mask out self-contrast cases
        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=self.device))

        # Compute denominator
        denominator = exp_sim.sum(dim=1, keepdim=True)

        # Compute log probabilities for positive pairs
        log_prob = similarity_matrix - torch.log(denominator + 1e-8)

        # Compute contrastive loss
        loss = (mask * log_prob).sum(dim=1) / pos_per_sample

        return -loss.mean()

    def train_contrastive(self, train_texts, train_labels, epochs=None, batch_size=64, gamma=0.5):
        # 使用类参数作为默认值
        if epochs is None:
            epochs = self.contrastive_epochs
        # 确保输入是Python列表（HuggingFace tokenizer对numpy数组支持有问题）
        if isinstance(train_texts, np.ndarray):
            train_texts = train_texts.tolist()
        elif isinstance(train_texts, pd.Series):
            train_texts = train_texts.astype(str).tolist()

        # 再次检查类型
        if not isinstance(train_texts, list):
            raise TypeError(f"train_texts must be list, got {type(train_texts)}")
        if not all(isinstance(x, str) for x in train_texts):
            raise TypeError("All elements in train_texts must be strings")

        # 转换标签
        train_labels = np.array(train_labels)
        y_train = torch.tensor(train_labels.astype(np.int64), device=self.device)

        # Tokenization（关键修改：显式转换为list）
        train_encodings = self.tokenizer(
            list(train_texts),  # 强制转换为Python list
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # 创建数据集
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            y_train
        )

        # Train labels array for index building
        train_labels_array = np.array(train_labels).astype(np.int64)

        # Create dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Train loop
        # FIX 2: Explicitly set models to train mode
        self.cl_encoder.train()
        self.bilstm.train()

        for epoch in range(epochs):
            total_loss = 0

            # Add tqdm progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                input_ids, attention_mask, labels = batch

                self.contrastive_optimizer.zero_grad()  # 原为 self.optimizer.zero_grad()

                # Get features from encoder
                features = self.cl_encoder(input_ids, attention_mask)

                # Compute contrastive loss
                con_loss = self.contrastive_loss_fn(features, labels)

                # Compute cross-entropy loss
                logits = self.bilstm(input_ids)
                ce_loss = self.ce_loss_fn(logits, labels)

                # Combined loss
                loss = ce_loss + gamma * con_loss

                loss.backward()
                self.contrastive_optimizer.step()

                total_loss += loss.item()

                # Update progress bar with current loss
                progress_bar.set_postfix({'loss': loss.item()})

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # After training, build index for retrieval
        self._build_retrieval_index(train_texts, train_labels_array)

    def _build_retrieval_index(self, train_texts, train_labels):
        """
        Build an approximate nearest neighbor index for fast retrieval
        """
        # FIX 3: Set all models to eval mode for index building
        self.cl_encoder.eval()
        self.bilstm.eval()

        # Initialize ANN index
        self.annoy_index = AnnoyIndex(self.vector_size, 'angular')

        # Encode all training samples with progress bar
        print("Building retrieval index...")
        progress_bar = tqdm(enumerate(train_texts), total=len(train_texts), desc="Encoding texts")

        for i, text in progress_bar:
            with torch.no_grad():
                vector = self.cl_encoder.encode_text(text, self.tokenizer, self.device)
                self.training_vectors.append(vector)
                self.training_labels.append(train_labels[i])
                self.annoy_index.add_item(i, vector)

        # Build the index
        self.annoy_index.build(15)  # 15 trees for better accuracy
        print("Retrieval index built successfully")

    def _retrieve_neighbors(self, text):
        """
        Retrieve k nearest neighbors for a text
        """
        # Encode query text
        query_vector = self.cl_encoder.encode_text(text, self.tokenizer, self.device)

        # Find nearest neighbors
        nn_indices = self.annoy_index.get_nns_by_vector(query_vector, self.k_neighbors)

        # Get corresponding vectors and labels
        nn_vectors = [self.training_vectors[i] for i in nn_indices]
        nn_labels = [self.training_labels[i] for i in nn_indices]

        return np.array(nn_vectors), np.array(nn_labels)

    def train_classifier(self, train_texts, train_labels, val_texts, val_labels, epochs=None, batch_size=64,
                         save_path="bilstm-am-best_model.pth"):
        # 使用类参数作为默认值
        if epochs is None:
            epochs = self.classifier_epochs
        # 确保输入是Python列表（HuggingFace tokenizer对numpy数组支持有问题）
        if isinstance(train_texts, np.ndarray):
            train_texts = train_texts.tolist()
        elif isinstance(train_texts, pd.Series):
            train_texts = train_texts.astype(str).tolist()

        # 再次检查类型
        if not isinstance(train_texts, list):
            raise TypeError(f"train_texts must be list, got {type(train_texts)}")
        if not all(isinstance(x, str) for x in train_texts):
            raise TypeError("All elements in train_texts must be strings")

        # 转换标签
        train_labels_array = np.array(train_labels).astype(np.int64)
        y_train = torch.tensor(train_labels_array, device=self.device)

        # Tokenization（显式转换为list）
        train_encodings = self.tokenizer(
            list(train_texts),  # 强制转换为Python list
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # Create tensors to store neighbor vectors and labels
        all_neighbor_vectors = []
        all_neighbor_labels = []

        # Retrieve neighbors for each training instance with progress bar
        print("Retrieving neighbors for training instances...")
        progress_bar = tqdm(train_texts, total=len(train_texts), desc="Retrieving neighbors")

        for text in progress_bar:
            nn_vectors, nn_labels = self._retrieve_neighbors(text)
            all_neighbor_vectors.append(nn_vectors)

            # Convert labels to one-hot
            one_hot_labels = np.zeros((self.k_neighbors, self.num_classes))
            for i, label in enumerate(nn_labels):
                one_hot_labels[i, label] = 1
            all_neighbor_labels.append(one_hot_labels)

        # Convert to tensors
        neighbor_vectors_tensor = torch.tensor(np.array(all_neighbor_vectors),
                                               dtype=torch.float32, device=self.device)
        neighbor_labels_tensor = torch.tensor(np.array(all_neighbor_labels),
                                              dtype=torch.float32, device=self.device)

        # Create dataset with neighbors
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            y_train,
            neighbor_vectors_tensor,
            neighbor_labels_tensor
        )

        # Create dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Train loop
        # FIX 4: Explicitly set all models to train mode
        self.model.train()
        self.bilstm.train()
        self.attention_mechanism.train()
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            self.bilstm.train()
            self.attention_mechanism.train()
            # Add tqdm progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                input_ids, attention_mask, labels, nn_vectors, nn_labels = batch

                self.classifier_optimizer.zero_grad()

                # Forward pass through model with attention mechanism
                outputs = self.model(input_ids, nn_vectors, nn_labels)

                # Compute loss
                loss = self.ce_loss_fn(outputs, labels)

                loss.backward()
                self.classifier_optimizer.step()

                total_loss += loss.item()

                # Update progress bar with current loss
                progress_bar.set_postfix({'loss': loss.item()})

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            # 验证步骤
            val_loss, val_f1 = self._evaluate_validation(val_texts, val_labels)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1} Validation || Loss: {val_loss:.4f} | F1: {val_f1:.4f}")

            # 保存最佳模型
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.best_model_state,
                    'f1': val_f1,
                }, save_path)
                print(f"New best model saved at epoch {epoch + 1} with F1 {val_f1:.4f}")

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # 训练结束后加载最佳模型
        self.model.load_state_dict(self.best_model_state)
        print(f"Loaded best model from epoch {self.best_epoch} with F1 {self.best_f1:.4f}")
        # 训练结束后绘制指标
        self._plot_training_losses(train_losses, val_losses)

    def _plot_training_losses(self, train_losses, val_losses):
        """绘制训练损失曲线"""
        epochs = len(train_losses)
        x = range(1, epochs + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(x, train_losses, 'b-', label='Training Loss')
        plt.plot(x, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(x)  # 根据实际epoch数设置刻度
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, texts):
        """
        Make predictions on new texts
        """
        # FIX 5: Explicitly set all models to eval mode for prediction
        self.model.eval()
        self.bilstm.eval()
        self.cl_encoder.eval()
        self.attention_mechanism.eval()

        predictions = []

        # Add progress bar for prediction
        progress_bar = tqdm(texts, total=len(texts), desc="Making predictions")

        for text in progress_bar:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Retrieve neighbors
            nn_vectors, nn_labels = self._retrieve_neighbors(text)

            # Convert to tensors
            nn_vectors_tensor = torch.tensor(nn_vectors, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Convert labels to one-hot
            one_hot_labels = np.zeros((self.k_neighbors, self.num_classes))
            for i, label in enumerate(nn_labels):
                one_hot_labels[i, label] = 1
            nn_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(inputs["input_ids"], nn_vectors_tensor, nn_labels_tensor)
                _, predicted = torch.max(outputs, 1)

            predictions.append(predicted.item())

        return predictions

    # 修改评估方法支持加载指定模型
    def evaluate(self, test_texts, test_labels, model_path=None):
        if model_path:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path} (Epoch {checkpoint['epoch']}, F1 {checkpoint['f1']})")

        # FIX 6: Explicitly set models to eval mode for evaluation
        self.model.eval()
        self.bilstm.eval()
        self.cl_encoder.eval()
        self.attention_mechanism.eval()

        # Convert test_labels to integers if needed
        test_labels_array = np.array(test_labels).astype(np.int64)

        print("Evaluating model on test set...")
        predictions = self.predict(test_texts)

        # Calculate metrics
        precision = precision_score(test_labels_array, predictions, average='macro') * 100
        recall = recall_score(test_labels_array, predictions, average='macro') * 100
        f1 = f1_score(test_labels_array, predictions, average='macro') * 100

        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        print(f"F1 Score: {f1:.2f}%")

        return precision, recall, f1





# Example usage
if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder

    # 初始化LabelEncoder
    label_encoder = LabelEncoder()

    # 数据预处理流程
    # 读取停用词文件
    def read_stopwords(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)


    # 过滤停用词
    def filter_stopwords(texts, stopwords):
        filtered_texts = []
        for text in texts:
            words = text.split()
            filtered_words = [word for word in words if word not in stopwords]
            filtered_text = ' '.join(filtered_words)
            filtered_texts.append(filtered_text)
        return filtered_texts


    # 数据加载
    train_df = pd.read_csv("./cnews.train.txt", sep='\t', names=['label', 'content'])
    test_df = pd.read_csv("./cnews.test.txt", sep='\t', names=['label', 'content'])
    val_df = pd.read_csv("./cnews.val.txt", sep='\t', names=['label', 'content'])

    X_train = train_df['content'].tolist()
    y_train = train_df['label']
    X_test = test_df['content'].tolist()
    y_test = test_df['label']
    X_val = val_df['content'].tolist()
    y_val = val_df['label']




    # 将标签转换为数字编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    from sklearn.model_selection import StratifiedShuffleSplit


    def sample_data(X, y, num_samples, random_state=None):
        rng = np.random.RandomState(random_state)
        unique_labels, label_counts = np.unique(y, return_counts=True)

        # 计算目标样本数时排除零样本类别
        valid_labels = [l for l, cnt in zip(unique_labels, label_counts) if cnt > 0]
        valid_counts = [cnt for cnt in label_counts if cnt > 0]
        label_proportions = valid_counts / sum(valid_counts)

        sample_sizes = (label_proportions * num_samples).astype(int)
        remaining = num_samples - sample_sizes.sum()

        # 剩余样本按比例分配
        while remaining > 0:
            for i in np.argsort(-label_proportions):
                sample_sizes[i] += 1
                remaining -= 1
                if remaining == 0:
                    break

        # 安全抽样（允许替换）
        sampled_indices = []
        for label, size in zip(valid_labels, sample_sizes):
            label_indices = np.where(y == label)[0]
            replace = len(label_indices) < size
            indices = rng.choice(label_indices, size=size, replace=replace)
            sampled_indices.extend(indices)

        # 确保总样本数正确
        if len(sampled_indices) > num_samples:
            sampled_indices = rng.choice(sampled_indices, num_samples, replace=False)
        elif len(sampled_indices) < num_samples:
            extra = num_samples - len(sampled_indices)
            all_indices = np.arange(len(y))
            extra_indices = rng.choice(all_indices, extra, replace=False)
            sampled_indices.extend(extra_indices)

        return [X[i] for i in sampled_indices], y[sampled_indices]


    # 采样
    X_train_sampled, y_train_sampled = sample_data(X_train, y_train_encoded, 5920, random_state=24)
    X_val_sampled, y_val_sampled = sample_data(X_val, y_val_encoded, 1080, random_state=24)
    X_test_sampled, y_test_sampled = sample_data(X_test, y_test_encoded, 1000, random_state=24)

    # 创建分类器时需要明确指定类别数量
    num_classes = len(label_encoder.classes_)  # 关键修改：获取实际类别数

    classifier = TextClassifier(
        num_classes=num_classes,  # 使用实际类别数
        contrastive_epochs=40,
        classifier_epochs=10
    )

    # 定义对比学习模型保存路径
    contrastive_model_path = "./models/bilstm-am-best-contrastive_model.pt"

    # 判断是否需要训练对比学习模型
    if os.path.exists(contrastive_model_path):
        print(f"发现已有对比学习模型: {contrastive_model_path}")
        loaded_successfully = classifier.load_contrastive_model(contrastive_model_path)

        if not loaded_successfully:
            print("加载对比学习模型失败，将重新训练...")
            # 训练并保存对比学习模型
            classifier.train_contrastive(
                X_train_sampled,
                y_train_sampled,
                epochs=10,
                batch_size=32
            )
            classifier.save_contrastive_model(contrastive_model_path)
    else:
        print("未找到已有对比学习模型，将进行训练...")
        # 训练并保存对比学习模型
        classifier.train_contrastive(
            X_train_sampled,
            y_train_sampled,
            epochs=10,
            batch_size=32
        )
        classifier.save_contrastive_model(contrastive_model_path)

    print("Training classifier with attention mechanism...")
    classifier.train_classifier(
        X_train_sampled,  # 使用采样后的训练数据
        y_train_sampled,
        val_texts=X_val_sampled,  # 使用采样后的验证数据
        val_labels=y_val_sampled,
        epochs=10,
        batch_size=32,
        save_path="bilstm-am-best_model.pth"
    )

    # 评估时使用采样后的测试数据
    print("Evaluating on test set...")
    precision, recall, f1 = classifier.evaluate(
        X_test_sampled,
        y_test_sampled,  # 使用采样后的测试数据
        model_path="bilstm-am-best_model.pth"
    )