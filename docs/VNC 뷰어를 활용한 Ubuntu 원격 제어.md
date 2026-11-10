# Ubuntu 원격 데스크톱 접속 가이드 — x11vnc + SSH

목적: SSH만 접속 가능한 원격 Ubuntu 컴퓨터에서, 기존에 실행 중인 GNOME 데스크톱 화면을 그대로 원격 제어한다.

XRDP와 달리 새로운 데스크톱 세션을 생성하지 않고, 물리 모니터에 표시되는 기존 세션을 공유한다.

## 1. 접속 구성

|항목|설정|
|---|---|
|원격 OS|Ubuntu / GNOME / Xorg|
|원격 사용자|`min`|
|원격 디스플레이|`:1`|
|X11 인증 경로|`/run/user/1000/gdm/Xauthority`|
|VNC 포트|`5900`|
|연결 방식|SSH 터널 + VNC Viewer|

위 설정은 이번에 확인한 원격 컴퓨터 기준이다. 다른 시스템이나 로그인 세션에서는 DISPLAY 번호와 인증 파일 경로가 달라질 수 있다.

## 2. 최초 1회 설치

### 원격 Ubuntu — x11vnc 설치

SSH로 원격 컴퓨터에 접속한다.

```
ssh min@원격컴퓨터_IP
```

x11vnc를 설치한다.

```
sudo apt update
sudo apt install -y x11vnc
```

### 내 Windows 컴퓨터 — VNC Viewer 설치

[RealVNC Viewer 다운로드](https://www.realvnc.com/en/connect/download/viewer/)에서 Windows용 Viewer를 설치한다.

다른 VNC 클라이언트를 사용해도 무방하다.

## 3. 매번 접속할 때 실행하는 명령

① 원격 Ubuntu에서 VNC 서버 실행

복사

```
x11vnc \
  -display :1 \
  -auth /run/user/1000/gdm/Xauthority \
  -localhost \
  -rfbport 5900 \
  -forever \
  -shared \
  -noxdamage
```

실행한 SSH 터미널을 유지한다. 정상 실행되면 VNC 서버가 5900 포트에서 연결을 기다린다.

② 내 Windows PC에서 SSH 터널 생성

복사

Windows PowerShell을 새로 열고 실행한다.

```
ssh -N -L 5900:127.0.0.1:5900 min@원격컴퓨터_IP
```

`원격컴퓨터_IP`는 실제 IP 주소로 교체한다. SSH 연결 후 아무 출력 없이 유지되는 것은 정상이다. 이 터미널도 닫지 않는다.

③ VNC Viewer로 접속

VNC Viewer 실행 후 아래 주소를 입력한다.

## 127.0.0.1:5900

복사

연결 성공 시 기존 Ubuntu GNOME 데스크톱이 표시된다. 듀얼 모니터도 같은 X11 데스크톱의 일부로 공유된다.

## 4. 명령어 옵션 설명

|옵션|기능|
|---|---|
|`-display :1`|기존 물리 모니터의 Xorg 디스플레이|
|`-auth ...`|기존 세션의 X11 인증 정보|
|`-localhost`|VNC 서버의 외부 직접 접속 제한|
|`-rfbport 5900`|VNC 접속 포트|
|`-forever`|VNC 클라이언트가 종료되어도 서버 유지|
|`-shared`|여러 VNC 클라이언트 접속 허용|
|`-noxdamage`|XDamage 기반 화면 갱신 비활성화|

SSH 터널을 사용하기 때문에 VNC 자체의 암호화에 의존하지 않는다. 다만 동일 PC의 다른 로컬 프로세스가 포트에 접근할 수 있으므로 신뢰할 수 있는 개인용 PC에서만 이 구성을 사용하고, 공유 PC에서는 VNC 인증을 추가하는 것이 좋다.

## 5. 접속이 안 될 때 확인할 명령

X11 디스플레이 확인

복사

X1이 존재하고 기존 GNOME 세션이 활성 상태인지 확인한다.

ls -la /tmp/.X11-unix/ loginctl list-sessions

VNC 서버 실행 여부

복사

x11vnc 프로세스 및 5900 포트 리스닝 여부를 확인한다.

pgrep -af x11vnc ss -ltn | grep 5900

실제 모니터 인식 확인

복사

HDMI-0, DP-3 등 실제 모니터가 connected로 표시되는지 확인한다.

DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority xrandr --query

VNC 서버 재실행

복사

기존 사용자 소유 x11vnc를 종료하고 같은 옵션으로 다시 실행한다.

pkill -u "$USER" -x x11vnc x11vnc -display :1 -auth /run/user/1000/gdm/Xauthority -localhost -rfbport 5900 -forever -shared -noxdamage

## 6. 접속 종료 방법

작업이 끝났다면 Windows의 VNC Viewer 연결을 종료하고 SSH 터널 터미널에서 `Ctrl+C`를 누른다.

VNC 서버까지 종료하려면 서버를 실행한 SSH 터미널에서 `Ctrl+C`를 누르면 된다.

## 핵심 요약

재접속용 최소 명령어

전체 복사

원격 Ubuntu

```
x11vnc -display :1 -auth /run/user/1000/gdm/Xauthority -localhost -rfbport 5900 -forever -shared -noxdamage
```

내 Windows PowerShell

```
ssh -N -L 5900:127.0.0.1:5900 min@원격컴퓨터_IP
```

VNC Viewer 접속 주소

`127.0.0.1:5900`

주의: 이 방식은 기존 Xorg 세션이 실행 중이어야 동작한다. 재부팅 후 로그인하지 않은 상태나 Wayland 세션에서는 동일한 명령으로 접속되지 않을 수 있다.


---

# Ubuntu 원격 데스크톱 복구 가이드 — 재부팅 후 GDM 로그인 화면에 VNC 연결

## 1. 목적

Ubuntu 컴퓨터를 재부팅한 이후, 기존에 사용하던 `x11vnc` 원격 데스크톱 연결이 실패하는 문제를 **물리적인 접근 없이 SSH만으로 해결**하는 방법을 설명한다.

특히 재부팅 직후 사용자가 GNOME에 로그인하지 않은 상태에서, **GDM 로그인 화면에 직접 VNC로 접속하여 원격 로그인을 수행하는 것**이 핵심이다.

기존 구성은 다음과 같다.

| 항목 | 설정 |
|---|---|
| 원격 OS | Ubuntu / GNOME / Xorg |
| 원격 사용자 | `irol` |
| VNC 서버 | `x11vnc` |
| VNC 포트 | `5900` |
| 접속 방식 | SSH 터널 + Remmina |
| 로그인 전 디스플레이 | `:0` (이번 사례) |
| 로그인 전 인증 파일 | `/run/user/128/gdm/Xauthority` (이번 사례) |

**주의:** 디스플레이 번호와 X11 인증 파일 경로는 부팅 또는 로그인 상태에 따라 달라질 수 있으므로 매번 확인하는 것이 좋다.

---

## 2. 문제 발생 원인

기존에는 다음 명령어로 VNC 서버를 실행했다.

```bash
x11vnc \
  -display :1 \
  -auth /run/user/1000/gdm/Xauthority \
  -localhost \
  -rfbport 5900 \
  -forever \
  -shared \
  -noxdamage
```

이 명령어는 이미 로그인되어 있는 GNOME 사용자 세션을 공유한다.

하지만 컴퓨터를 재부팅하면 기존 사용자 세션이 종료되고, Ubuntu는 GDM 로그인 화면으로 진입한다.

이때 디스플레이 구성이 다음과 같이 달라질 수 있다.

| 상태 | 로그인 전 | 로그인 후 |
|---|---|---|
| 화면 | GDM 로그인 화면 | GNOME 사용자 데스크톱 |
| 실행 사용자 | `gdm` | `irol` |
| DISPLAY | `:0` | `:1` 등 |
| XAUTHORITY | GDM 인증 파일 | 사용자 세션 인증 파일 |

즉, 재부팅 직후에는 기존의 `:1` 디스플레이가 존재하지 않을 수 있다.

또한 로그인 화면은 `gdm` 사용자가 관리하기 때문에 일반 사용자 권한으로는 해당 화면의 인증 파일을 읽지 못할 수 있다.

따라서 **로그인 화면의 Xorg 디스플레이 및 인증 파일을 확인하고, 해당 세션에 x11vnc를 연결해야 한다.**

---

## 3. 복구 절차

### STEP 1. SSH 접속

클라이언트 컴퓨터에서 원격 Ubuntu에 접속한다.

```bash
ssh irol@원격컴퓨터_IP
```

이후 명령은 별도 설명이 없는 한 원격 Ubuntu에서 실행한다.

### STEP 2. 현재 Xorg 디스플레이 확인

```bash
ls -la /tmp/.X11-unix/
```

실제 발생했던 사례:

```text
srwxrwxrwx 1 gdm gdm 0 Oct 21 18:40 X0
```

여기에서 중요한 것은 `X0`이다.

이는 X11 디스플레이 `:0`에 해당하는 소켓이 존재한다는 의미다.

또한 소켓의 소유자가 `gdm`이므로 GDM 로그인 화면에서 사용 중일 가능성이 높다.

단, 소켓 소유자만으로 현재 로그인 상태를 확정할 수는 없다.

### STEP 3. GDM의 X11 인증 파일 확인

**이번 문제 해결에서 가장 중요한 과정이다.**

다음 명령어를 실행한다.

```bash
ps -efww | grep '[X]org'
```

실제 출력:

```text
gdm 1294 1287 0 18:40 tty1 00:00:00 /usr/lib/xorg/Xorg vt1 -displayfd 3 -auth /run/user/128/gdm/Xauthority -nolisten tcp -background none -noreset -keeptty -novtswitch -verbose 3
```

여기에서 다음 정보를 추출한다.

| 항목 | 값 |
|---|---|
| Xorg 실행 사용자 | `gdm` |
| Xorg 인증 파일 | `/run/user/128/gdm/Xauthority` |
| 디스플레이 번호 | `:0` (STEP 2에서 확인) |

특히 다음 부분이 중요하다.

```text
-auth /run/user/128/gdm/Xauthority
```

이 경로는 **Xorg 프로세스가 실제로 사용 중인 인증 파일**이다.

기존의 `/run/user/1000/gdm/Xauthority`를 그대로 사용하지 말고, 현재 Xorg 프로세스에 지정된 경로를 사용한다.

### STEP 4. GDM 로그인 화면에 x11vnc 연결

위에서 확인한 정보를 바탕으로 실행한다.

```bash
sudo x11vnc \
  -display :0 \
  -auth /run/user/128/gdm/Xauthority \
  -localhost \
  -rfbport 5900 \
  -forever \
  -shared \
  -noxdamage \
  -noshm
```

이번 사례에서는 이 구성을 통해 GDM 로그인 화면에 접근할 수 있었다.

옵션 설명:

| 옵션 | 기능 |
|---|---|
| `sudo` | GDM의 X11 인증 파일에 접근할 권한 확보 |
| `-display :0` | 로그인 화면이 실행 중인 Xorg 디스플레이 |
| `-auth ...` | GDM에서 사용하는 X11 인증 파일 |
| `-localhost` | VNC 서버의 외부 직접 접속 제한 |
| `-rfbport 5900` | VNC 포트 지정 |
| `-forever` | VNC 클라이언트 연결 종료 후에도 서버 유지 |
| `-shared` | 여러 클라이언트 접속 허용 |
| `-noxdamage` | XDamage 기반 화면 갱신 비활성화 |
| `-noshm` | MIT-SHM 공유 메모리 사용 비활성화 |

`-noshm`은 이번 사례에서 추가로 필요했던 옵션이다.

해당 옵션이 없으면 다음 오류가 발생했다.

```text
X11 MIT Shared Memory Attach failed

X Error of failed request:
  BadAccess (attempt to access private resource denied)

Major opcode of failed request: 130 (MIT-SHM)
Minor opcode of failed request: 1 (X_ShmAttach)
```

이는 화면 캡처 과정에서 MIT-SHM 공유 메모리 접근에 실패한 것이다.

`-noshm` 옵션을 사용하면 공유 메모리 기반 캡처를 비활성화하여 해당 오류를 우회할 수 있다.

정상적으로 실행되면 다음 로그가 나타난다.

```text
Using X display :0
Listening for VNC connections on TCP port 5900
```

VNC 프로세스가 종료되지 않고 실행 상태를 유지해야 한다.

### STEP 5. 클라이언트에서 SSH 터널 생성

클라이언트 Ubuntu의 별도 터미널에서 실행한다.

```bash
ssh -N -L 5900:127.0.0.1:5900 irol@원격컴퓨터_IP
```

이미 로컬 5900 포트를 다른 원격 컴퓨터에서 사용하고 있다면 5901 등으로 변경한다.

```bash
ssh -N -L 5901:127.0.0.1:5900 irol@원격컴퓨터_IP
```

포트 포워딩 구조:

```text
클라이언트 Ubuntu
    │
    │ 127.0.0.1:5901
    │
    ▼
SSH 터널
    │
    │ 암호화된 SSH 연결
    │
    ▼
원격 Ubuntu
    │
    │ 127.0.0.1:5900
    │
    ▼
x11vnc → GDM 로그인 화면
```

### STEP 6. Remmina로 로그인 화면 접속

Remmina에서 다음과 같이 설정한다.

| 항목 | 값 |
|---|---|
| 프로토콜 | VNC |
| 서버 | `127.0.0.1:5900` 또는 `127.0.0.1:5901` |
| SSH 터널 | 비활성화 (외부 SSH 터널 사용 시) |

정상적으로 연결되면 원격 Ubuntu의 GDM 로그인 화면이 나타난다.

여기에서 `irol` 계정으로 로그인한다.

**로그인 성공 이후에는 일반적인 GNOME 데스크톱 세션이 생성된다.**

---

## 4. 로그인 이후 주의사항

GDM 로그인 화면과 일반 사용자 GNOME 세션은 서로 다른 Xorg 디스플레이를 사용할 수 있다.

예를 들어 다음과 같다.

```text
로그인 전
  GDM
  DISPLAY=:0
  XAUTHORITY=/run/user/128/gdm/Xauthority

          ↓ 로그인

로그인 후
  GNOME (irol)
  DISPLAY=:1
  XAUTHORITY=/run/user/1000/gdm/Xauthority
```

위 로그인 후 값은 예시이며 실제 시스템에서 확인해야 한다.

로그인 과정에서 기존 GDM 세션이 종료되면 VNC 연결도 끊어질 수 있다.

이 경우 다음 명령어로 새로운 디스플레이와 인증 경로를 확인한다.

```bash
ls -la /tmp/.X11-unix/
ps -efww | grep '[X]org'
```

이후 기존 x11vnc를 종료하고 일반 사용자 데스크톱에 다시 연결한다.

예시:

```bash
x11vnc \
  -display :1 \
  -auth /run/user/1000/gdm/Xauthority \
  -localhost \
  -rfbport 5900 \
  -forever \
  -shared \
  -noxdamage \
  -noshm
```

일반 사용자 권한으로 인증 파일에 접근할 수 없다면 경로 및 권한을 확인하고 필요한 경우 `sudo`를 사용한다.

클라이언트 측 SSH 터널과 Remmina 설정은 그대로 유지할 수 있다.

---

## 5. 주요 오류와 해결 방법

| 오류 | 원인 | 해결 |
|---|---|---|
| `XOpenDisplay failed (:1)` | `:1` 디스플레이가 존재하지 않음 | 실제 DISPLAY 번호 확인 |
| `-auth guess: failed` | X11 인증 파일 자동 탐색 실패 | Xorg 프로세스의 `-auth` 경로 직접 지정 |
| `MIT Shared Memory Attach failed` | MIT-SHM 공유 메모리 접근 실패 | `-noshm` 추가 |
| `Address already in use` | 해당 포트가 이미 사용 중 | 기존 프로세스 확인 또는 포트 변경 |
| VNC 접속 후 로그인 화면만 표시 | GDM 화면에 연결된 상태 | GNOME 로그인 후 사용자 세션에 재연결 |
| `Type=wayland` | Wayland 세션 사용 | Xorg 세션 사용 여부 확인 |

참고로 IPv6 바인딩에서만 `Address already in use`가 발생하고 IPv4의 5900 포트는 정상적으로 리스닝하고 있다면, 기존 IPv4 SSH 터널 연결에는 문제가 없을 수 있다.

### Wayland 관련

현재 가이드는 Xorg를 전제로 한다.

Wayland에서는 x11vnc가 동일한 방식으로 데스크톱을 공유할 수 없다.

현재 세션 유형은 다음 명령으로 확인할 수 있다.

```bash
loginctl list-sessions
```

세션 ID를 확인한 뒤:

```bash
loginctl show-session 세션ID -p Type -p Class -p Name
```

`Type=x11`이면 Xorg 세션이며, `Type=wayland`이면 별도의 대응이 필요하다.

---

## 6. 재발 시 최소 복구 명령어

다음 순서대로 진행한다.

**① SSH 접속**

```bash
ssh irol@원격컴퓨터_IP
```

**② 디스플레이 및 인증 파일 확인**

```bash
ls -la /tmp/.X11-unix/
ps -efww | grep '[X]org'
```

**③ GDM 로그인 화면에 x11vnc 연결**

이번 환경에서 확인한 설정:

```bash
sudo x11vnc \
  -display :0 \
  -auth /run/user/128/gdm/Xauthority \
  -localhost \
  -rfbport 5900 \
  -forever \
  -shared \
  -noxdamage \
  -noshm
```

단, `:0`과 `/run/user/128/gdm/Xauthority`는 반드시 현재 환경과 일치하는지 확인한다.

**④ 클라이언트에서 SSH 터널 생성**

```bash
ssh -N -L 5900:127.0.0.1:5900 irol@원격컴퓨터_IP
```

**⑤ Remmina로 접속**

```text
VNC → 127.0.0.1:5900
```

**⑥ GNOME 로그인 후 연결이 끊어지면**

새로운 사용자 Xorg 세션의 DISPLAY와 XAUTHORITY를 확인하여 x11vnc를 재실행한다.

---

## 7. 보안 및 운영상 주의사항

현재 명령어에는 VNC 비밀번호 인증이 설정되어 있지 않다.

`-localhost`와 SSH 터널을 사용하므로 외부 네트워크에서 VNC 포트에 직접 접속하는 것은 제한되지만, 원격 호스트 내부의 다른 사용자나 프로세스가 접근할 가능성까지 차단하지는 않는다.

특히 GDM 로그인 화면을 `sudo` 권한의 x11vnc로 공유하는 경우에는 접근 통제에 주의해야 한다.

장기적으로 운영할 경우에는 VNC 비밀번호 인증을 추가하고, SSH 계정 접근 권한을 제한하는 것이 바람직하다.

또한 `sudo`로 실행한 x11vnc는 높은 권한을 가진 프로세스이므로 필요한 동안에만 실행하도록 한다.

---

## 핵심 정리

**재부팅 후 x11vnc 접속이 실패하는 이유는 기존 GNOME 사용자 세션이 사라지고 GDM 로그인 화면만 남아 있기 때문일 수 있다.**

이때 핵심적인 해결 과정은 다음과 같다.

1. `/tmp/.X11-unix/`에서 실제 디스플레이 번호를 확인한다.
2. `ps -efww | grep '[X]org'`로 GDM이 사용하는 X11 인증 파일을 확인한다.
3. `sudo x11vnc`에 해당 DISPLAY와 XAUTHORITY 경로를 명시한다.
4. MIT-SHM 접근 오류가 발생하면 `-noshm`을 추가한다.
5. SSH 터널과 Remmina를 통해 GDM 로그인 화면에 접속한다.
6. GNOME 로그인 후 필요하면 새로운 사용자 세션에 x11vnc를 다시 연결한다.

**가장 중요한 명령어는 `ps -efww | grep '[X]org'`이다.**

이 명령어를 통해 현재 Xorg가 실제로 사용하는 인증 파일을 확인할 수 있으며, 해당 정보를 기반으로 로그인 전 GDM 화면에 직접 연결할 수 있다.