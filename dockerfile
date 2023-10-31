FROM public.ecr.aws/lambda/python:3.8

# Copy function code and models into our /var/task
COPY ./ ${LAMBDA_TASK_ROOT}/
COPY ./model ${LAMBDA_TASK_ROOT}/model

# install our dependencies
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# run faaslight
RUN mv ${LAMBDA_TASK_ROOT}/faaslight /var
WORKDIR /var/faaslight
RUN python3 -m pip install -r requirements.txt

# avoiding problems 
RUN rm ${LAMBDA_TASK_ROOT}/joblib/test/test_func_inspect_special_encoding.py
RUN sed -i '247d' ${LAMBDA_TASK_ROOT}/caffe2/python/layers/layers.py

# RUN python3 integrationFunMain.py

WORKDIR ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handler" ]