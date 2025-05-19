import torch.nn as nn
from torch import device
from transformers import BertTokenizer
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt



# 数据加载
train_df = pd.read_csv("./cnews.train.txt", sep='\t', names=['label', 'content'])
test_df = pd.read_csv("./cnews.test.txt", sep='\t', names=['label', 'content'])
val_df = pd.read_csv("./cnews.val.txt", sep='\t', names=['label', 'content'])

X_train = train_df['content'].tolist()
y_train = train_df['label'].tolist()
X_test = test_df['content'].tolist()
y_test = test_df['label'].tolist()
X_val = val_df['content'].tolist()
y_val = val_df['label'].tolist()


# 采样函数
def sample_data(texts, labels, num_samples, random_state=None):
    rng = np.random.RandomState(random_state)
    # 将标签转换为数组
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    # 计算采样数量
    label_counts = np.array([np.sum(labels == label) for label in unique_labels])
    label_proportions = label_counts / np.sum(label_counts)
    sample_sizes = (label_proportions * num_samples).astype(int)
    remaining = num_samples - np.sum(sample_sizes)
    sample_sizes[np.argsort(label_proportions)[-remaining:]] += 1

    # 采样索引
    sampled_indices = []
    for label, size in zip(unique_labels, sample_sizes):
        indices = np.where(labels == label)[0]
        selected = rng.choice(indices, size=size, replace=False)
        sampled_indices.extend(selected)

    # 打乱顺序避免类别顺序偏差
    rng.shuffle(sampled_indices)

    # 返回采样后的文本和标签
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    return sampled_texts, sampled_labels



# 训练集采样
X_train_sampled, y_train_sampled = sample_data(X_train, y_train, 5920, random_state=24)
# 验证集采样
X_val_sampled, y_val_sampled = sample_data(X_val, y_val, 1080, random_state=24)
# 测试集采样
X_test_sampled, y_test_sampled = sample_data(X_test, y_test, 1000, random_state=24)

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('./MiniRBT-h256-pt')

# 编码函数
def encode_texts(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

# 编码采样后的数据
X_train_bert = encode_texts(X_train_sampled)
X_val_bert = encode_texts(X_val_sampled)
X_test_bert = encode_texts(X_test_sampled)

# 标签编码（在采样后进行）
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_sampled)  # 注意使用采样后的标签
y_val_encoded = label_encoder.transform(y_val_sampled)
y_test_encoded = label_encoder.transform(y_test_sampled)

# === 后续代码保持不变（只需调整变量名）===
# 转换标签为张量
train_labels = torch.tensor(y_train_encoded, dtype=torch.long)
val_labels = torch.tensor(y_val_encoded, dtype=torch.long)
test_labels = torch.tensor(y_test_encoded, dtype=torch.long)

# 创建数据集（使用采样后的编码数据）
train_dataset = TensorDataset(
    X_train_bert['input_ids'],
    X_train_bert['attention_mask'],
    train_labels
)
val_dataset = TensorDataset(
    X_val_bert['input_ids'],
    X_val_bert['attention_mask'],
    val_labels
)
test_dataset = TensorDataset(
    X_test_bert['input_ids'],
    X_test_bert['attention_mask'],
    test_labels
)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = tokenizer.vocab_size
print(vocab_size)
num_classes = len(set(y_train))


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, num_classes, dropout_prob=0.5):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

        conved = [self.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]  # 全局最大池化
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        output = self.fc(cat)
        return output

# 通用评估函数
def evaluate(model, dataloader, criterion, description="Evaluating"):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    eval_iterator = tqdm(dataloader, desc=description, unit="batch")
    with torch.no_grad():
        for batch in eval_iterator:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    return avg_loss, precision, recall, f1

# 超参数设置
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 128
dropout_prob = 0.5
lr = 1e-3

# 初始化CNN模型
cnn_model = CNNModel(vocab_size, embedding_dim, filter_sizes, num_filters, num_classes, dropout_prob)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
cnn_criterion = nn.CrossEntropyLoss()
# 训练CNN模型
scheduler = StepLR(cnn_optimizer, step_size=5, gamma=1)
train_losses = []  # 训练损失记录
val_losses = []  # 验证损失记录
num_epochs = 10
for epoch in range(num_epochs):
    cnn_model.train()
    total_loss = 0
    train_iterator = tqdm(train_dataloader, desc=f"CNN Epoch {epoch + 1}/10", unit="batch")
    for batch in train_iterator:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        cnn_optimizer.zero_grad()
        outputs = cnn_model(input_ids)
        loss = cnn_criterion(outputs, labels)
        loss.backward()
        cnn_optimizer.step()
        total_loss += loss.item()
        train_iterator.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)  # 记录训练损失
    val_loss, val_precision, val_recall, val_f1 = evaluate(cnn_model, val_dataloader, cnn_criterion, 'CNN Validation')
    val_losses.append(val_loss)  # 记录验证损失
    scheduler.step()

    print(f"Epoch {epoch + 1}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
# ==================== 绘制损失曲线 ====================
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, 'r-^', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 最终测试评估
test_loss, test_precision, test_recall, test_f1 = evaluate(cnn_model, test_dataloader, cnn_criterion, 'CNN Testing')
print('=== Test Results ===')
print(f'Test Loss: {test_loss:.4f}')
print(f'Precision: {test_precision:.4f}')
print(f'Recall: {test_recall:.4f}')
print(f'F1 Score: {test_f1:.4f}')


