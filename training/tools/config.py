import json

DEFAULTS = {
    "network": "dpn",
    "encoder": "dpn92",
    "model_params": {},
    "optimizer": {
        "batch_size": 32,
        "type": "SGD",  # supported: SGD, Adam
        "momentum": 0.9,
        "weight_decay": 0,
        "clip": 1.,
        "learning_rate": 0.1,
        "classifier_lr": -1,
        "nesterov": True,
        "schedule": {
            "type": "constant",  # supported: constant, step, multistep, exponential, linear, poly
            "mode": "epoch",  # supported: epoch, step
            "epochs": 10,
            "params": {}
        }
    },
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}

RSNA_CFG = {'image_target_cols': ['pe_present_on_image'],
 'exam_target_cols': ['negative_exam_for_pe',
  'rv_lv_ratio_gte_1',
  'rv_lv_ratio_lt_1',
  'leftsided_pe',
  'chronic_pe',
  'rightsided_pe',
  'acute_and_chronic_pe',
  'central_pe',
  'indeterminate'],
 'image_weight': 0.07361963,
 'exam_weights': [0.0736196319,
  0.2346625767,
  0.0782208589,
  0.06257668712,
  0.1042944785,
  0.06257668712,
  0.1042944785,
  0.1877300613,
  0.09202453988]}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config