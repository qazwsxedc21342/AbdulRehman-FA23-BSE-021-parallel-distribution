import socket

HOST = "127.0.0.1"
PORT = 5000

# Create TCP socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to server
client.connect((HOST, PORT))

print("=" * 40)
print("TCP CLIENT")
print("=" * 40)
print("Connected to server.")
print("Type 'exit' to close the connection.\n")


while True:
    message = input("You: ")

    if message.lower() == "exit":
        break

    client.send(message.encode())

    response = client.recv(1024).decode()

    print("Server:", response)


client.close()
print("Disconnected from server.")