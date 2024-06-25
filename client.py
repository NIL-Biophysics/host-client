import socket


def is_binary_data(data):
    return data.startswith((b'\x89PNG', b'\xFF\xD8\xFF', b'GIF87a', b'GIF89a'))


def send_image(image_path, prefix_byte='AA'):
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()

    if prefix_byte.upper() == 'AA':
        extra_byte = b'\xAA'
    elif prefix_byte.upper() == 'AB':
        extra_byte = b'\xAB'
    else:
        raise ValueError("prefix_byte must be either 'AA' or 'AB'")

    img_bytes = extra_byte + img_bytes

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 50000))

    client_socket.sendall(img_bytes)
    client_socket.shutdown(socket.SHUT_WR)

    response = b''
    while True:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        response += chunk

    client_socket.close()

    if is_binary_data(response):
        with open('received_mask.png', 'wb') as f:
            f.write(response)
        print("Received mask image saved as 'received_mask.png'")
    else:
        try:
            print("Received:", response.decode('utf-8'))
        except UnicodeDecodeError:
            print("Received data could not be decoded as UTF-8. Raw data:", response)


if __name__ == "__main__":
    send_image('microscope_image_112.png', prefix_byte='AB')

