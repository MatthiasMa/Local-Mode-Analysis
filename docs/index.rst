# Local-Mode-Analysis
## decoding IR-spectra by visualizing molecular details



>>> pip install lma

Local-Mode-Analysis
*************

:class:`~LMA` .

See `XXX <apidoc/XXXConsumer.html>`_ for API and configuration details.

The consumer iterator returns ConsumerRecords, which are simple namedtuples
that expose basic message attributes: topic, partition, offset, key, and value:

>>> from XXX import XXXConsumer
>>> consumer = XXXConsumer('my_favorite_topic')
>>> for msg in consumer:
...     print (msg)

>>> # manually assign the partition list for the consumer
>>> from XXX import TopicPartition
>>> consumer = XXXConsumer(bootstrap_servers='localhost:1234')
>>> consumer.assign([TopicPartition('foobar', 2)])
>>> msg = next(consumer)

>>> # Deserialize msgpack-encoded values
>>> consumer = XXXConsumer(value_deserializer=msgpack.loads)
>>> consumer.subscribe(['msgpackfoo'])
>>> for msg in consumer:
...     assert isinstance(msg.value, dict)


XXXProducer
*************

:class:`~XXX.XXXProducer` is a high-level, asynchronous message producer.
The class is intended to operate as similarly as possible to the official java
client. See `XXXProducer <apidoc/XXXProducer.html>`_ for more details.


Compression
***********



Legacy support is maintained for low-level consumer and producer classes,
SimpleConsumer and SimpleProducer.


.. toctree::
   :hidden:
   :maxdepth: 2

   Usage Overview <usage>
   API </apidoc/modules>
   Simple Clients [deprecated] <simple>
   install
   tests
   compatibility
   support
   license
   changelog
