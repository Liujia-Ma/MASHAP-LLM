import argparse
import sys
import numbers
sys.path.append("../../")
from mashap_llm.mas import MAS
from mashap_llm.eval.evaluators import MathEvaluator, CodingEvaluator

EVALUATOR_MAP = {
    "math": MathEvaluator,
    "coding": CodingEvaluator,
}

def build_parser():
    parser = argparse.ArgumentParser(description='universal entry for evaluation')

    parser.add_argument('--evaluator_type', type=str, required=True, choices=EVALUATOR_MAP.keys(), help='the type of evaluator, e.g. math')
    parser.add_argument('--model_path', type=str, required=True, help='path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='path to the data')
    parser.add_argument('--profile_path', type=str, default=None, help='path to the profile')
    parser.add_argument('--load_path', type=str, default=None, help='path to adapter/checkpoint to load')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory')
    parser.add_argument('--response_filename', type=str, default=None, help='response file name')
    parser.add_argument('--metrics_filename', type=str, default=None, help='metrics file name')
    parser.add_argument('--metrics_timestamp', action='store_true', help='add timestamp to metrics file name')
    parser.add_argument('--eval_seed', type=int, default=None, help='seed for shuffled evaluation order; if not set, shuffle with random seed')

    # Generation parameters
    parser.add_argument('--num_agents', type=int, default=1, help='number of agents')
    parser.add_argument('--context_window', type=int, default=2048, help='context window size')
    parser.add_argument('--top_k', type=int, default=50, help='top k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p sampling')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for sampling')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='maximum number of new tokens')
    parser.add_argument('--do_sample', dest='do_sample', action='store_true', default=True, help='enable sampling (default: True)')
    parser.add_argument('--no_do_sample', dest='do_sample', action='store_false', help='disable sampling')
    
    # 动态添加评估器特定参数    
    subparsers = parser.add_subparsers(dest='subcommand')
    for eval_name, eval_cls in EVALUATOR_MAP.items():
        sub_parser = subparsers.add_parser(eval_name)
        eval_cls.add_args(sub_parser)
    
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    evaluator_cls = EVALUATOR_MAP[args.evaluator_type]

    mas = MAS(**vars(args), skip_critic_init=True)
    if args.load_path is not None:
        print(f"✅ Successfully loaded adapter from path: {args.load_path}")
    else:
        print("⚠️ Load path is not provided; evaluating base model only.")
    
    evaluator = evaluator_cls(
        mas=mas,
        data_path=args.data_path,
        output_dir=args.output_dir,
        metrics_filename=args.metrics_filename,
        metrics_timestamp=args.metrics_timestamp,
        response_filename=args.response_filename,
        eval_seed=args.eval_seed,
    )
    
    # 执行评估
    metrics = evaluator.evaluate()
    
    # 输出结果
    print("\n评估结果:")
    for k, v in metrics.items():
        if isinstance(v, numbers.Real):
            print(f"{k}: {float(v):.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == '__main__':
    main()
