# Start from cuda image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set timezone and non-interactive frontend
RUN apt-get update && apt-get install -y wget sudo vim tmux htop git nano pigz parallel curl apt-transport-https ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y google-cloud-cli libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*


# Set working directory to home
WORKDIR /root

# Copy and install requirements only
COPY requirements.txt .

# Install pip dependencies within environment
RUN pip install -r requirements.txt

# Then copy the rest of the code
COPY . .

# Enter bash once to initialize everything
CMD ["/bin/bash", "-l"]

