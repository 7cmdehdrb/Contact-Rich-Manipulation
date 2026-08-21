````md
# Ubuntu Server에서 PPTP VPN 연결하기

## 1. 필요한 패키지 설치

```bash
sudo apt update
sudo apt install pptp-linux ppp
````

## 2. PPTP 연결 설정 생성

```bash
sudo pptpsetup \
  --create <연결이름> \
  --server <PPTP서버주소> \
  --username <사용자이름> \
  --password '<비밀번호>' \
  --encrypt
```

예시:

```bash
sudo pptpsetup \
  --create irol_up \
  --server irol.iptime.org \
  --username irol \
  --password '<PASSWORD>' \
  --encrypt
```

설정 파일은 다음 위치에 생성된다.

```text
/etc/ppp/peers/irol_up
```

계정 정보는 다음 파일에 등록된다.

```text
/etc/ppp/chap-secrets
```

## 3. PPTP 연결

```bash
sudo pon irol_up
```

연결 확인:

```bash
ip addr show ppp0
```

또는:

```bash
ip a
```

정상적으로 연결되면 `ppp0` 인터페이스가 생성된다.

## 4. 라우팅 확인

```bash
ip route
```

특정 IP가 어느 인터페이스를 사용하는지 확인:

```bash
ip route get <목적지_IP>
```

예:

```bash
ip route get 192.168.50.44
```

## 5. PPTP 연결 종료

```bash
sudo poff irol_up
```

## 6. 연결 문제 확인

최근 PPP 로그:

```bash
plog
```

실시간으로 자세히 확인:

```bash
sudo pppd call irol_up nodetach debug
```

## 핵심 명령만 요약

```bash
# 최초 1회 설정
sudo pptpsetup \
  --create irol_up \
  --server irol.iptime.org \
  --username irol1 \
  --password 'PWD' \
  --encrypt

# 연결
sudo pon irol_up

# 확인
ip addr show ppp0

# 종료
sudo poff irol_up
```

# 로그인 정보 기입

```bash
export OMNI_USER="omniverse"
export OMNI_PASS="PWD"
```

# Route 추가

```bash
sudo ip route add 192.168.0.0/24 dev ppp0
```