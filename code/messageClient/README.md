# Dealing with the python message client

This script initiates the communication client and manages the corresponding AI reguests.

The tool was originally developed by Dr.-Ing. Marcus Grum.

## Getting-Started

1. Start the `message broker`. Further details can be found at the corresponding `Readme.md`.

1. Start the `messaging client` by

    ```
    python3 AI_simulation_basis_communication_client.py
    ```

1. Initiate a request, which for instance can come from an Industry 4.0 production system, a modeling software or manually.

	```
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=realize_annExperiment, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "localhost" -p 1883
    ```

## AI Requests

### 1. wire_annSolution

### 2. create_annSolution

### 3. refine_annSolution

### 4. apply_annSolution

### 5. evaluate_annSolution

### 6. realize_annExperiment
