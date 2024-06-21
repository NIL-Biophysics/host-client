import socket


def send_image(image_path):
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 50000))

    client_socket.sendall(img_bytes)
    client_socket.shutdown(socket.SHUT_WR)

    response = client_socket.recv(1024)
    print("Received:", response.decode())
    client_socket.close()


if __name__ == "__main__":
    send_image('microscope_image_112.png')
