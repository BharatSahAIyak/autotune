[![Coverage Status](https://coveralls.io/repos/github/ChakshuGautam/AutoTuneNLP/badge.svg?branch=main)](https://coveralls.io/github/ChakshuGautam/AutoTuneNLP?branch=main)

# AutoTuneNLP
A comprehensive toolkit for seamless data generation and fine-tuning of NLP models, all conveniently packed into a single block.

# Setup

Clone the repo and cd to project root.

## Environment
1. Create and activate venv. Ex:
(on windows)
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## API
1. Start your docker engine and run a redis image on port 6379.
```
docker run --name autotunenlp-redis -p 6379:6379 -d redis
```
2. Start celery worker.
```
celery -A autotune worker --loglevel=info
```
- If you are running on windows, the above command won't work since celery is not supported on windows, but you can use the below command for testing (caveat: it's capabilities are lost).
```
celery -A autotune worker --loglevel=info  --pool=solo
```
3. Specify a port number and start the application.
```
uvicorn autotune.asgi:application --port PORT_NUMBER --reload
```

## Contributing
Interested in contributing to AutoTune? We'd love your help! Check out our [issues section](https://github.com/BharatSahAIyak/autotune/issues) for areas where you can contribute. Please see our [contribution guide](CONTRIBUTION.md) for more details on how to get involved.


## Typical Workflow
0. User is shown a login page to login using their Google account. (The account has to be of 'samagragovernance.in' domain). The user is then shown a settings page where they are nudged to update the API keys for OpenAI and HuggingFace. The user is also shown a list of all the repos they have access to on HuggingFace. The user can select the repo they want to work on and the settings are saved. The `settings` tab allows the user to view and update the settings.
1. User gives a prompt and samples are 5 generated automatically.
2. User can view the 5 generated samples and select the ones they like and dislike.
3. The prompt is updated with those examples and the process is repeated until the user is satisfied with the 5 samples.
4. Once the user is satified, user provides the number of samples they want and the data is generated. The process is async and progress is shown to the user. The progress is tracked every 2 seconds.
5. The data is generated and shown to the user, they give a go ahead and the dataset is pushed to huggingface. The `dataset` tab allows the user to view all the datasets that they have deployed until now on huggingface.
6. A link is shared with the user so that they can view the data on huggingface.
7. There is a tab called train, which allows user to use the dataset created earlier to train a model by filling a form. The process is async and progress is shown to the user. The progress is tracked every 2 seconds.
8. Once trained the user is allowed to view the results of the model on the test set. The user can also view the results of the model on the validation set. The `models` tab allows the user to view all the models that they have deployed until now.
9. The user is then nudged to deploy the model to huggingface. Once confirmed, the user is asked to provide a name for the model and the model is pushed to huggingface.
10. A link is shared with the user so that they can view the model on huggingface and a curl is shared so that they can use the model for inference.
11. The `history` tab allows the user to view all the tasks that they have performed until now.



## License

AutoTune is made available under the [MIT License](LICENSE). See the [LICENSE](https://opensource.org/licenses/MIT) file for more info.

Thank you for considering AutoTune for your machine learning and dataset creation needs. We're excited to see the innovative solutions you'll build with our platform!

