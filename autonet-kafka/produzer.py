from pykafka import KafkaClient
import time

client = KafkaClient("127.0.0.1:9092")
training = client.topics["training"]

with training.get_sync_producer() as producer:
    i = 0
    for _ in range(10):
        producer.produce(("Kafka is not just anotther " + str(i)).encode("ascii"))
        i += 1
        time.sleep(1)
