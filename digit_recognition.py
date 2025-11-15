import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# ----------------------------
# 1. Define the same CNN model
# ----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------------
# 2. Load the trained model
# ----------------------------
model = CNNModel()
model.load_state_dict(torch.load("mnist_cnn_model.pth"))
model.eval()

# ----------------------------
# 3. Define Transform
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# 4. Open Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
print("Press 'S' to save snapshot, 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (28, 28))
    roi = cv2.bitwise_not(roi)  # invert colors if needed
    tensor = transform(roi).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        digit = predicted.item()

    cv2.putText(frame, f"Prediction: {digit}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Digit Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("snapshot.png", frame)
        print("Snapshot saved!")

cap.release()
cv2.destroyAllWindows()
