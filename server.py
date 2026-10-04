import socket
import threading

HOST = "127.0.0.1"
PORT = 5000

# Lock for thread synchronization
lock = threading.Lock()


def handle_client(client_socket, client_address):
    thread_name = threading.current_thread().name

    print(f"\n[NEW CONNECTION]")
    print(f"Active Thread Name: {thread_name}")
    print(f"Client IP: {client_address[0]}")
    print(f"Port Number: {client_address[1]}")

    try:
        while True:
            message = client_socket.recv(1024).decode()

            if not message:
                break

            print(f"[{thread_name}] Client: {message}")

            # Synchronization using Lock
            lock.acquire()

            try:
                response = f"Server received: {message}"
                client_socket.send(response.encode())
            finally:
                lock.release()

    except ConnectionResetError:
        print(f"[{thread_name}] Client disconnected unexpectedly.")

    finally:
        client_socket.close()
        print(f"[{thread_name}] Connection closed.")


# Create TCP socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind server with IP and port
server.bind((HOST, PORT))

# Start listening for clients
server.listen(5)

print("=" * 50)
print("MULTI-THREADED TCP SERVER")
print("=" * 50)
print(f"Server started on {HOST}:{PORT}")
print("Waiting for clients...\n")


while True:
    client_socket, client_address = server.accept()

    # Create a new thread for every client
    client_thread = threading.Thread(
        target=handle_client,
        args=(client_socket, client_address)
    )

    client_thread.start()

    print(f"Active Threads: {threading.active_count()}")