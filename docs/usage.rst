Usage
*****


LMA
=============

.. code:: python

    from lma import whatever

    # To consume latest messages and auto-commit offsets
    x = xConsumer('my-topic',
                             group_id='my-group',
                             bootstrap_servers=['localhost:9092'])



There are many...


LMAX
==============

.. code:: python

    from lma import lmax
    

