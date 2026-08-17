# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=22.04
ARG DXVK_VERSION=2.6.2

FROM ubuntu:${UBUNTU_VERSION} AS dxvk
ARG DXVK_VERSION
ARG DXVK_SHA256=17761876556afd55736cb895d184f5a1c55d43350f1b1e3b129f8d28706d7992
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils
RUN curl --fail --location --retry 3 \
      --output /tmp/dxvk.tar.gz \
      "https://github.com/doitsujin/dxvk/releases/download/v${DXVK_VERSION}/dxvk-${DXVK_VERSION}.tar.gz" && \
    echo "${DXVK_SHA256}  /tmp/dxvk.tar.gz" | sha256sum --check --strict && \
    mkdir -p /opt/dxvk && \
    tar --extract --gzip --file /tmp/dxvk.tar.gz --directory /opt/dxvk --strip-components=1

FROM ubuntu:${UBUNTU_VERSION} AS runtime-base

ARG WINEHQ_VERSION=11.0.0.0~jammy-1
ARG WINETRICKS_VERSION=20250102
ARG WINETRICKS_SHA256=53194dead910f8a5eb1deacaa4773d4e48f5873633d18ab1ecd6fdb0cb92243b

ENV DEBIAN_FRONTEND=noninteractive \
    DSLE_ASSET_DIR=/opt/dsle/assets \
    DSLE_CONTAINER=1 \
    DSLE_DESKTOP_RES=800x600 \
    DSLE_GAME_DIR=/opt/dsle/game \
    DSLE_INSTANCE_CONFIG=/var/lib/dsle/config/instances.json \
    DSLE_RUNTIME_SOCKET=/var/lib/dsle/run/dsle.sock \
    DSLE_RUNTIME_TOKEN_FILE=/var/lib/dsle/run/dsle.token \
    DSLE_STATE_DIR=/var/lib/dsle \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PULSE_SERVER=unix:/var/lib/dsle/pulse/native \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/dsle/src \
    PYTHONUNBUFFERED=1 \
    WINEARCH=win64 \
    WINEDEBUG=-all \
    WINEPREFIX=/opt/dsle/wine-template \
    WINEPREFIX_TEMPLATE=/opt/dsle/wine-template \
    WINEDLLOVERRIDES="winemenubuilder.exe=d;mscoree,mshtml=" \
    XDG_RUNTIME_DIR=/var/lib/dsle/xdg/container

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      cabextract \
      ca-certificates \
      curl \
      fonts-wine \
      gnupg2 \
      libasound2 \
      libasound2:i386 \
      libgl1 \
      libgl1:i386 \
      libpulse0 \
      libpulse0:i386 \
      libvulkan1 \
      libvulkan1:i386 \
      libx11-6 \
      libx11-6:i386 \
      libxext6 \
      libxext6:i386 \
      p7zip-full \
      procps \
      psmisc \
      pulseaudio \
      python3 \
      python3-pip \
      util-linux \
      vulkan-tools \
      winbind \
      x11-utils \
      x11-xserver-utils \
      x11vnc \
      xauth \
      xserver-xorg-core \
      xserver-xorg-video-dummy \
      xvfb

RUN mkdir -p /etc/apt/keyrings && \
    curl --fail --silent --show-error --location \
      https://dl.winehq.org/wine-builds/winehq.key \
      | gpg --dearmor --output /etc/apt/keyrings/winehq-archive.key && \
    curl --fail --silent --show-error --location \
      https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources \
      | sed 's#Signed-By:.*#Signed-By: /etc/apt/keyrings/winehq-archive.key#' \
      > /etc/apt/sources.list.d/winehq-jammy.sources && \
    apt-get update && \
    apt-get install -y --install-recommends "winehq-stable=${WINEHQ_VERSION}"

RUN curl --fail --location --retry 3 \
      --output /usr/local/bin/winetricks \
      "https://raw.githubusercontent.com/Winetricks/winetricks/${WINETRICKS_VERSION}/src/winetricks" && \
    echo "${WINETRICKS_SHA256}  /usr/local/bin/winetricks" | sha256sum --check --strict && \
    chmod 0755 /usr/local/bin/winetricks

COPY docker/container_requirements.txt /opt/dsle/container_requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==26.2.1 setuptools==80.9.0 wheel==0.45.1 && \
    python3 -m pip install --no-cache-dir \
      --requirement /opt/dsle/container_requirements.txt

RUN set -eux; \
    mkdir -p /opt/dsle/wine-template /var/lib/dsle; \
    export DISPLAY=:99; \
    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp -noreset > /tmp/xvfb-prefix.log 2>&1 & \
    xpid=$!; \
    trap 'kill -TERM "${xpid}" 2>/dev/null || true; wait "${xpid}" 2>/dev/null || true' EXIT; \
    wineboot --init; \
    WINETRICKS_SUPER_QUIET=1 winetricks --unattended --force win10 vcrun2022 d3dcompiler_47; \
    wineserver --wait

COPY --from=dxvk /opt/dxvk/x64/d3d11.dll /opt/dsle/wine-template/drive_c/windows/system32/d3d11.dll
COPY --from=dxvk /opt/dxvk/x64/dxgi.dll /opt/dsle/wine-template/drive_c/windows/system32/dxgi.dll
COPY --from=dxvk /opt/dxvk/x32/d3d11.dll /opt/dsle/wine-template/drive_c/windows/syswow64/d3d11.dll
COPY --from=dxvk /opt/dxvk/x32/dxgi.dll /opt/dsle/wine-template/drive_c/windows/syswow64/dxgi.dll

RUN set -eux; \
    export DISPLAY=:99; \
    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp -noreset > /tmp/xvfb-dxvk.log 2>&1 & \
    xpid=$!; \
    trap 'kill -TERM "${xpid}" 2>/dev/null || true; wait "${xpid}" 2>/dev/null || true' EXIT; \
    wine reg add 'HKCU\Software\Wine\DllOverrides' /v d3d11 /t REG_SZ /d native,builtin /f; \
    wine reg add 'HKCU\Software\Wine\DllOverrides' /v dxgi /t REG_SZ /d native,builtin /f; \
    wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f; \
    wineserver --wait

LABEL org.opencontainers.image.title="DSLE NVIDIA runtime" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.description="Wine/DXVK/X11 runtime; game files are supplied at run time" \
      org.opencontainers.image.licenses="GPL-3.0-only" \
      io.dsle.dxvk.version="2.6.2" \
      io.dsle.wine.version="11.0"

WORKDIR /opt/dsle
EXPOSE 5901-5930
VOLUME ["/var/lib/dsle"]
ENTRYPOINT ["/usr/local/bin/dsle-entrypoint"]
CMD ["dsle-server"]

# Research dependencies branch directly from the stable Wine/core dependency
# filesystem. Source, asset, example, and entrypoint edits therefore cannot
# invalidate this large CUDA/PyTorch layer.
FROM runtime-base AS research-dependencies

RUN python3 -m pip install --no-cache-dir \
      cma==3.4.0 \
      scipy==1.15.3 && \
    python3 -m pip install --no-cache-dir \
      torch==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu128

# Assemble application-owned files in an isolated staging root. Both final
# targets copy this exact payload; no Wine prefix or dependency tree is copied
# between stages.
FROM runtime-base AS application-payload

COPY LICENSE /tmp/dsle-payload/opt/dsle/LICENSE
COPY src /tmp/dsle-payload/opt/dsle/src
COPY assets /tmp/dsle-payload/opt/dsle/assets
RUN cd /tmp/dsle-payload/opt/dsle && \
    sha256sum --check --strict --quiet assets/checksums.sha256
COPY examples /tmp/dsle-payload/opt/dsle/examples
COPY docker/xorg/nvidia.conf.template /tmp/dsle-payload/opt/dsle/xorg/nvidia.conf.template
COPY --chmod=0755 docker/bin/dsle-doctor /tmp/dsle-payload/usr/local/bin/dsle-doctor
COPY --chmod=0755 docker/bin/dsle-health /tmp/dsle-payload/usr/local/bin/dsle-health
COPY --chmod=0755 docker/bin/dsle-instances /tmp/dsle-payload/usr/local/bin/dsle-instances
COPY --chmod=0755 docker/bin/dsle-server /tmp/dsle-payload/usr/local/bin/dsle-server
COPY --chmod=0755 docker/entrypoint.sh /tmp/dsle-payload/usr/local/bin/dsle-entrypoint

FROM runtime-base AS runtime

COPY --from=application-payload /tmp/dsle-payload/ /

LABEL io.dsle.image.flavor="core"

FROM research-dependencies AS research

COPY --from=application-payload /tmp/dsle-payload/ /

LABEL io.dsle.image.flavor="research" \
      io.dsle.pytorch.version="2.7.1+cu128"
