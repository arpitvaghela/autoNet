from pykafka import KafkaClient
import os

bstrap_server = dict(os.environ)["DOCKER_GATEWAY_HOST"]
client = KafkaClient(hosts=bstrap_server)


def get_messages(topicname):
    def events():
        for message in client.topics[topicname].get_simple_consumer():
            yield message.value.decode()

    return events()


for x in get_messages("log"):
    print(x)
