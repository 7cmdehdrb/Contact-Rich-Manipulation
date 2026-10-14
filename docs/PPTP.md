# Ubuntu PPTP VPN 연결

기존 PPTP VPN 서버에 접속할 때 사용하는 명령을 정리한다. 서버 주소·계정·연결 이름은
사용하는 VPN의 값으로 바꾼다.

## 최초 설정

```bash
sudo apt update
sudo apt install pptp-linux ppp
sudo pptpsetup \
  --create vpn_connection \
  --server <VPN_SERVER> \
  --username <USERNAME> \
  --password '<PASSWORD>' \
  --encrypt
```

연결 설정은 `/etc/ppp/peers/vpn_connection`, 계정 설정은 `/etc/ppp/chap-secrets`에 생성된다.

## 연결과 종료

```bash
sudo pon vpn_connection
ip addr show ppp0
ip route
sudo poff vpn_connection
```

## 라우팅과 연결 점검

```bash
ip route get <DESTINATION_IP>
plog
sudo pppd call vpn_connection nodetach debug
```

VPN 연결 후 내부 네트워크 경로가 추가로 필요한 구성에서는 해당 주소 범위와 인터페이스를
확인해 route를 추가한다. 아래 주소 범위는 예시다.

```bash
sudo ip route add 192.168.0.0/24 dev ppp0
```

[일반 지식 안내](README.md)로 돌아간다.
