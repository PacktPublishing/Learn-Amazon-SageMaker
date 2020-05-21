FROM r-base:latest

RUN R -e "install.packages(c('rjson', 'plumber'), repos='https://cloud.r-project.org')"

COPY main.R /opt/ml/
COPY train_function.R /opt/ml/
COPY serve_function.R /opt/ml/

WORKDIR /opt/ml/

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/main.R", "--no-save"]