`ip addr show wlo1 | grep -o "inet [0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*"`

docker script

to run compose use

if running controller locally

```
export DOCKER_GATEWAY_HOST=$(hostname -I |awk '{print $1}')
export SYSTEM_IP=$(hostname -I |awk '{print $1}')
sudo -E docker-compose up
```

if running controller and kafka elsewhere

```
export DOCKER_GATEWAY_HOST= <YOUR IP ADDRESS>
export SYSTEM_IP=$(hostname -I |awk '{print $1}')
sudo -E docker-compose up
```
