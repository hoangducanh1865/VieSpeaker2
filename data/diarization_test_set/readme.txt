conda create -n label-studio-spk-diarization python==3.10
conda activate label-studio-spk-diarization
pip install label-studio

---- dựng label studio và ngrok ----
conda activate label-studio-spk-diarization
export LOCAL_FILES_SERVING_ENABLED=true
export ALLOWED_HOSTS="localhost,127.0.0.1,belen-monadelphous-unverbosely.ngrok-free.dev"
export CSRF_TRUSTED_ORIGINS="https://*.ngrok-free.dev,https://*.ngrok-free.app"
label-studio start --host 0.0.0.0 --port 8082
ngrok http 8082

---- Tạo 1 campaign (không có preannotation) ----
0. Chuẩn bị:
    - folder source-storage
    - file json (không cần tạo file json này nếu đánh giá tất cả mọi audio trong folder source-storage)
    - interface.html
1. Ấn nút tạo campaign
2. Nhập tên campaign và mô tả của campaign
3. Sang tab upload file json, up file json lên.
Format file json: tham khảo file mẫu: projects/nhap/nhap.json
4. Sang tab interface, copy paste interface.html vào
5. Ấn Save để hoàn thành
6. Vào Setting -> Vào cloud -> Add source storage
   Chọn Local file --> Chọn cái format image, audio, video --> Nhập abs path của folder source-storage
   Ấn add source storage.
   Không ấn sync. Chỉ save thôi.
=> Done

----- zrok -----
Cách install:
    mkdir -p ~/downloads/zrok && cd ~/downloads/zrok
    curl -L -o zrok.tgz "https://github.com/openziti/zrok/releases/download/v1.1.10/zrok_1.1.10_linux_amd64.tar.gz"
    (muốn tìm đúng file bin cần tải đúng với máy thì dùng:
        uname -m # nếu là x86_64 thì chọn x86_64
    sau đó lấy url tương ứng trên này: https://docs.zrok.io/docs/guides/install/linux/
    )
    tar -xzf zrok.tgz
    mkdir -p ~/bin
    install -m 0755 zrok ~/bin/zrok
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

Verify lại:
    which zrok
    zrok version

Lấy auth token và add token:
    Truy cập vào zrok. Đăng ký tài khoản
    Đăng ký xong tài khoản sẽ hiện ra 1 dòng: zrok enable <auth_token>
    Add token vào hệ thống bằng cách chạy dòng zrok enable trên
    Verify: zrok status

Cách chạy/dựng zrok:
    zrok share public <port> (VD: zrok share public 8082)
