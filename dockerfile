FROM ubuntu:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
     gcc-aarch64-linux-gnu \
     g++-aarch64-linux-gnu \
     cmake \
     make \
     wget \
     tar \
     unzip \
     build-essential \
     && rm -rf /var/lib/apt/lists/*

RUN wget https://sdk.lunarg.com/sdk/download/1.4.321.1/linux/vulkansdk-linux-x86_64-1.4.321.1.tar.xz -O vulkansdk.tar.xz \
     && tar -xf vulkansdk.tar.xz -C /usr \
     && mv /usr/1.4.321.1 /usr/vulkan_sdk \
     && rm vulkansdk.tar.xz

RUN wget https://dl.google.com/android/repository/android-ndk-r27d-linux.zip -O android-ndk.zip \
    && unzip android-ndk.zip -d /usr \
    && rm android-ndk.zip

ENV VULKAN_SDK=/usr/vulkan_sdk/x86_64
ENV PATH=$VULKAN_SDK/bin:$PATH:/usr/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin
ENV VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
ENV Vulkan_INCLUDE_DIR=$VULKAN_SDK/include
ENV Vulkan_LIBRARY=$VULKAN_SDK/lib/libvulkan.so

RUN apt-get update && apt-get install -y curl libcurl4-openssl-dev

# Set default shell
SHELL ["/bin/bash", "-c"]