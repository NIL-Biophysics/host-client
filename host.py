# PART 0 - IMPORT PACKAGES
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import socket
import io


# PART 1 - MODEL
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs


# PART 2 - PRED-PROCESSING
def split(picture):
    width, height = picture.size
    size = (width, height)
    piece_width = width // 3
    piece_height = height // 2
    pieces = []
    for y in range(2):
        for x in range(3):
            left = x * piece_width
            upper = y * piece_height
            right = left + piece_width
            lower = upper + piece_height
            piece = picture.crop((left, upper, right, lower))
            resized = piece.resize((416, 416))
            pieces.append(resized)
    return pieces, size


# PART 3 - MODEL USAGE
def predict(list):
    checkpoint_path = "retina_unet.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    mask_list = []
    for i in range(len(list)):
        image = np.array(list[i])
        image = image[:, :, ::-1].copy()
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        with torch.no_grad():
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8) * 255
        mask_list.append(pred_y)
    return mask_list


# PART 4 - POST-PROCESSING
def labeling(image):
    mask = clear_border(image)
    mask_labeled = label(mask)
    return mask_labeled


def process_image(image):
    mask_labeled = labeling(image)

    def func(slc):
        new_slc = np.zeros_like(slc)
        rps = regionprops(slc)
        filtered_rps = [r for r in rps if r.area >= 80]
        for j, r in enumerate(filtered_rps):
            new_slc[tuple(r.coords.T)] = j + 1
        return new_slc

    mask_labeled = func(mask_labeled)
    mask = mask_labeled > 0
    mask_filled = ndi.binary_fill_holes(mask)
    return mask_filled


def process_list(mask_list):
    processed_list = []
    for img in mask_list:
        processed_img = process_image(img)
        processed_list.append(processed_img)
    return processed_list


# PART 5 - REUNITE
def reunite(mask_list, size):
    img_height, img_width = mask_list[0].shape
    combined_image = np.zeros((img_height * 2, img_width * 3), dtype=np.uint8)
    combined_image[0:img_height, 0:img_width] = mask_list[0]
    combined_image[0:img_height, img_width:img_width * 2] = mask_list[1]
    combined_image[0:img_height, img_width * 2:img_width * 3] = mask_list[2]
    combined_image[img_height:img_height * 2, 0:img_width] = mask_list[3]
    combined_image[img_height:img_height * 2, img_width:img_width * 2] = mask_list[4]
    combined_image[img_height:img_height * 2, img_width * 2:img_width * 3] = mask_list[5]
    combined_image = cv2.resize(combined_image, size, interpolation=cv2.INTER_AREA)
    return combined_image


# PART 6 - COORDINATES
def coordinates(image):
    blur = cv2.GaussianBlur(image, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 40, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (255, 0, 255), 1)
    coord_list = []
    for contour in cnt:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(2 * M["m10"] / M["m00"])
            cy = int(2 * M["m01"] / M["m00"])
            coord_list.append((cx, cy))
    return coord_list


# PART 7 - MAIN FUNCTION (for testing without GUI)
def process_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    subpics, original_size = split(img)
    predicted = predict(subpics)
    post_processed = process_list(predicted)
    mask = reunite(post_processed, original_size)
    mask = mask * 255
    cv2.imwrite('y_pred.png', mask)
    coords = coordinates(mask)
    for i, coord in enumerate(coords):
        print(f"Cell {i + 1}: {coord[0]}, {coord[1]}")
    return coords


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8080))
    server_socket.listen(5)
    print("Server listening on port 8080...")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        # Receiving data in chunks
        data = b""
        while True:
            packet = client_socket.recv(1024 * 1024)
            if not packet:
                break
            data += packet

        if data:
            coords = process_image_from_bytes(data)
            response = f"Coordinates: {coords}"
            client_socket.send(response.encode())

        client_socket.close()


if __name__ == "__main__":
    start_server()
