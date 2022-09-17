import mlflow


def start_logging(cfg):
    if cfg['train']['log_metrics']:
        mlflow.start_run(run_name=cfg['train']['experiment_name'])


def end_logging(cfg):
    if cfg['train']['log_metrics']:
        mlflow.end_run()


def log_metrics(names, metrics, step, cfg):
    if cfg['train']['log_metrics']:
        for name, metric in zip(names, metrics):
            mlflow.log_metric(name, metric, step)


def log_params(cfg):
    if cfg['train']['log_metrics']:
        mlflow.log_param('cfg', cfg)
