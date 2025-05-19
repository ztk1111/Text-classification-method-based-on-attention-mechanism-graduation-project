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



# ... [保持其他导入和参数不变] ...

# 数据加载（保持不变）
train_df = pd.read_csv("./cnews.train.txt", sep='\t', names=['label', 'content'])
test_df = pd.read_csv("./cnews.test.txt", sep='\t', names=['label', 'content'])
val_df = pd.read_csv("./cnews.val.txt", sep='\t', names=['label', 'content'])

X_train = train_df['content'].tolist()  # 改为列表形式
y_train = train_df['label'].tolist()
X_test = test_df['content'].tolist()
y_test = test_df['label'].tolist()
X_val = val_df['content'].tolist()
y_val = val_df['label'].tolist()




# 修改后的采样函数（处理原始文本列表）
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


# === 核心修改：先采样再处理 ===
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


# 定义BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        return output

# 评估BiLSTM模型
def bilstm_evaluate(dataloader, criterion, description="BiLSTM Evaluating"):
    bilstm_model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    eval_iterator = tqdm(dataloader, desc=description, unit="batch")
    with torch.no_grad():
        for batch in eval_iterator:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            bilstm_outputs = bilstm_model(input_ids, attention_mask)
            loss = criterion(bilstm_outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(bilstm_outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    return avg_loss, precision, recall, f1

embedding_dim =256
hidden_dim = 128
num_layers = 2
lr = 1e-3

# 使用最优超参数重新训练模型并在测试集上评估
bilstm_model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bilstm_model.to(device)
bilstm_optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=lr,weight_decay=1e-4)
bilstm_criterion = nn.CrossEntropyLoss()

# 定义学习率调度器
step_size = 5  # 每5个epoch调整一次学习率
gamma = 1 # 学习率衰减因子
scheduler = StepLR(bilstm_optimizer, step_size=step_size, gamma=gamma)
train_losses = []  # 记录训练损失
val_losses = []    # 记录验证损失
num_epochs = 20
# 训练BiLSTM模型
for epoch in range(num_epochs):
    bilstm_model.train()
    bilstm_total_loss = 0
    bilstm_train_iterator = tqdm(train_dataloader, desc=f"BiLSTM Epoch {epoch + 1}/10", unit="batch")
    for batch in bilstm_train_iterator:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        bilstm_optimizer.zero_grad()
        bilstm_outputs = bilstm_model(input_ids, attention_mask)
        bilstm_loss = bilstm_criterion(bilstm_outputs, labels)
        bilstm_loss.backward()
        bilstm_optimizer.step()
        bilstm_total_loss += bilstm_loss.item()
        bilstm_train_iterator.set_postfix(loss=bilstm_loss.item())

    # 计算平均训练损失
    avg_train_loss = bilstm_total_loss / len(train_dataloader)

    # 评估验证集
    val_loss, val_precision, val_recall, val_f1 = bilstm_evaluate(val_dataloader, bilstm_criterion, 'BiLSTM Validation')



    # 更新学习率
    scheduler.step()
    # 记录本epoch的损失
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(
        f"  Validation Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")
    print(f"  Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

# 在训练结束后绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs+1))  # 显示所有epoch刻度
plt.legend()
plt.grid(True)
# 保存并显示图像
# plt.savefig('./loss_curve.png')  # 保存为图片文件
plt.show()                      # 显示窗口

# 评估测试集
test_loss, test_precision, test_recall, test_f1 = bilstm_evaluate(test_dataloader, bilstm_criterion,
'BiLSTM Testing')
print('=== Final Test Results ===')
print(f'Test Loss: {test_loss:.4f}')
print(f'Precision: {test_precision:.4f}')
print(f'Recall: {test_recall:.4f}')
print(f'F1 Score: {test_f1:.4f}')