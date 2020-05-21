FROM r-base:latest

RUN R -e "install.packages(c('rjson'), repos='https://cloud.r-project.org')"

COPY train.R /opt/ml/

WORKDIR /opt/ml/

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/train.R", "--no-save"]