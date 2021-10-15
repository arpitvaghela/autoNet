`ip addr show wlo1 | grep -o "inet [0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*"`

docker script

to run compose use

```
export DOCKER_GATEWAY_HOST=$(hostname -I |awk '{print $1}')
sudo -E docker-compose up
```
