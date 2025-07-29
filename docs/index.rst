HF Eco2AI Plugin Documentation
==============================

A Hugging Face Trainer callback that logs CO₂, kWh, and regional grid intensity for every epoch.
Built on Eco2AI's best-in-class energy tracking.

Features
--------

- **Seamless HF integration** - Just add one callback
- **Real-time carbon tracking** - CO₂ emissions per epoch/step  
- **Regional grid data** - Accurate carbon intensity by location
- **Prometheus export** - For production monitoring
- **Beautiful dashboards** - Grafana templates included

Quick Start
-----------

.. code-block:: python

   from transformers import Trainer
   from hf_eco2ai import Eco2AICallback

   trainer = Trainer(
       model=model,
       args=training_args,
       callbacks=[Eco2AICallback()]
   )
   
   trainer.train()

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`