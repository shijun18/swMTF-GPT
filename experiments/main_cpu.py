"""GPT-like model in Mesh-Tensorflow"""

from functools import partial
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from utils import save_config, expand_attention_types_params, yes_or_no, remove_gs_or_filepath, setup_logging
from inputs import sequential_input, mlm_sample_text, generic_text
from export import export_model
from model_fns_cpu import model_fn
from configs import fetch_model_params
import argparse
import json
import numpy
import os

cluster = {'chief': ['localhost:2224']}


# cluster = {'chief': ['localhost:2223'],
#            'worker': ['localhost:2223','localhost:2223', 'localhost:2223']
#           }

# cluster = {'chief': ['172.19.12.71:2224'],
#            'worker': ['172.19.12.72:2224','172.19.12.73:2224', '172.19.12.74:2224']
#           }

# cluster = {'chief': ['localhost:2222'],
#             'ps': ['localhost:2222'],
#             'worker': ['localhost:2222', 'localhost:2222']
#           }


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, default=None, help="Name of TPU to train on, if any.")
    parser.add_argument("--gpu_ids", nargs="+", type=str, default=None,
                        help="If training on GPU, can specify your GPU names in a list - i.e 'device:GPU:0 device:GPU:1'")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=5000, help="Save a model checkpoint every X steps.")
    parser.add_argument("--auto_layout", action="store_true", help="If set, generates and prints the most memory "
                                                                   "efficient layout according to MTF auto layout.")
    parser.add_argument("--auto_layout_and_mesh_shape", action="store_true",
                        help="If set, generates and prints the most memory efficient layout and mesh shape according to"
                             " MTF auto layout.")
    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")
    
    parser.add_argument("--entmax_sampling", action="store_true", help="(experimental) use entmax sampling")
    parser.add_argument("--export", action="store_true", help="If set, will export the model.")
    parser.add_argument("--task", type=str, help="current role in the distributed training")
    parser.add_argument("--index", type=int, help="current index in the process group")
    args = parser.parse_args()
    assert args.model is not None, "Model must be set"
    return args


def main(args):
    # Setup logging
    logger = setup_logging(args)

    # Read params of model
    params = fetch_model_params(args.model)
    print(params)

    # Fetch appropriate input functions
    
    input_fn = params.get("input_fn", "sequential_input")
    if input_fn == "sequential_input":
        input_fn = sequential_input
    elif input_fn == "generic_text":
        input_fn = generic_text

    # get current step
    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logger.info(f"Current step {current_step}")

    if current_step is None:
        current_step = 1

    if params["mlm_training"]:
        mlm_sample_text_fn = partial(mlm_sample_text, params)
        input_fn = partial(generic_text, sample_text_fn=mlm_sample_text_fn)

    # Confirm deletion of checkpoint files if --new flag is set
    if args.new and args.task == 'chief':
        if yes_or_no(f"Are you sure you want to remove '{params['model_path']}' to start afresh?"):
            remove_gs_or_filepath(params["model_path"])
        else:
            exit()

    # Save config to logdir for experiment management
    if args.task == 'chief':
        save_config(params, params["model_path"])

    # Add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    params["num_cores"] = mesh_shape.size
    params["auto_layout"] = args.auto_layout
    params["auto_layout_and_mesh_shape"] = args.auto_layout_and_mesh_shape
    params["use_tpu"] = True if not args.tpu is None else False
    params["gpu_ids"] = args.gpu_ids
    params["steps_per_checkpoint"] = args.steps_per_checkpoint
    # Expand attention types param
    params["attention_types"] = expand_attention_types_params(params["attention_types"])
    assert len(params["attention_types"]) == params["n_layer"]  # Assert that the length of expanded list = num layers
    params['model'] = params.get("model", "GPT") # Default model selection to GPT since it's the only option for now
    params["export"] = args.export
    # Set sampling parameters
    params["sampling_use_entmax"] = args.entmax_sampling

    # Sample quality of MoE models suffers when using the faster sampling method, so default to slow_sampling if
    # moe layers are present
    params["slow_sampling"] = True if params["moe_layers"] is not None else False

    logger.info(f"params = {params}")


    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    config = tf.estimator.RunConfig(
        model_dir=params["model_path"],
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=params["iterations"],
        save_summary_steps=params["iterations"],
        train_distribute=strategy
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params)
    

    if args.export and args.task == 'chief':
        export_model(estimator, "export", params)
        return

    # just train
    while current_step < params["train_steps"]:
        # Else, don't stop and restart
        train_spec = tf.estimator.TrainSpec(input_fn=partial(input_fn, global_step=current_step, eval=False), max_steps=1000)
        eval_spec = tf.estimator.EvalSpec(input_fn=partial(input_fn, global_step=current_step, eval=False))
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        

    
if __name__ == "__main__":
    tf.disable_v2_behavior()
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster,
                                          'task': {'type': args.task, 'index': args.index}})
    main(args)
