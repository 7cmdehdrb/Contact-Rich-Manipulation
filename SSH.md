



## Ubuntu에서 SSH 서버 열기

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable --now ssh
```

상태 확인:

```bash
sudo systemctl status ssh
```

방화벽을 사용 중이라면 SSH 허용:

```bash
sudo ufw allow OpenSSH
sudo ufw status
```

접속할 Ubuntu PC의 IP 확인:

```bash
hostname -I
```

다른 PC에서 접속:

```bash
ssh 사용자이름@Ubuntu_IP
```

예:

```bash
ssh min@192.168.0.10
```

외부 인터넷에서 접속하려면 공유기에서 **TCP 22번 포트 포워딩**도 설정해야 합니다. 보안을 위해 외부 공개 시에는 키 인증과 포트 변경을 권장합니다.