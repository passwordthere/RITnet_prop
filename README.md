```bash
# 下载dataset
wget https://dl.min.io/client/mc/release/darwin-arm64/mc    # mac ARM64 (Apple Silicon)
wget https://dl.min.io/client/mc/release/windows-amd64/mc.exe   # windows AMD64 (x64)
wget https://dl.min.io/client/mc/release/linux-amd64/mc     # linux AMD64 (x64)
# 其他系统查找 https://dl.min.io/client/mc/release/

git clone git@github.com:passwordthere/RITnet_prop.git
cd RITnet_prop
mc alias set myminio http://sclera.synology.me:19000 baiyi baiyi2022
mc cp myminio/tao/mock_camera_od.tar.gz .
tar -xzvf mock_camera_od.tar.gz
python3 testrealtime_task_outerconer_pulse_qt.py    # 需先启动QT软件
```