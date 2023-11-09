import pathlib
import os

from verifier import analyze
from utils.loading import parse_spec
from networks import get_network
from time import perf_counter

from rich import print


DEVICE = 'cpu'
PROJECT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), '..')


def get_gt():
    gt_file = os.path.join(PROJECT_PATH, 'test_cases', 'gt.txt')
    with open(gt_file) as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        content = line.split(',')
        model_type, img, result = tuple(content)
        if model_type not in data:
            data[model_type] = {}
        data[model_type][img] = result.startswith('verified')
    return data


def main():
    score = 0
    max_score = 0
    gt = get_gt()

    for i, net_name in enumerate(gt):
        for j, spec in enumerate(gt[net_name]):
            gt_verified = gt[net_name][spec]
            true_label, dataset, image, eps = parse_spec(os.path.join(PROJECT_PATH, 'test_cases', net_name, spec))
            net = get_network(net_name,
                              dataset, os.path.join(PROJECT_PATH, f'models/{dataset}_{net_name}.pt')).to(DEVICE)

            image = image.to(DEVICE)
            out = net(image.unsqueeze(0))

            pred_label = out.max(dim=1)[1].item()
            assert pred_label == true_label

            start = perf_counter()
            verified = analyze(net, image, eps, true_label)
            dt = perf_counter() - start

            print('[white]--------------------------------------------------')
            print(f'[white]{i+1}.{j+1}  {net_name}: {spec}')
            test_score = 0
            if verified and not gt_verified:
                test_score = -2
                print(f'  verified      [red]wrong    [white]([red]{test_score}[white])  ({dt:.3f}s)')
            elif verified and gt_verified:
                test_score = 1
                max_score += 1
                print(f'  verified      [green]correct  [white]([green]+{test_score}[white])  ({dt:.3f}s)')
            elif not verified and gt_verified:
                max_score += 1
                print(f'  not verified  [yellow]wrong    [white]([yellow]+{test_score}[white])  ({dt:.3f}s)')
            else:
                print(f'  not verified  [green]correct  [white]([green]+{test_score}[white])  ({dt:.3f}s)')
            score += test_score

    print('[white]--------------------------------------------------')
    print(f'\n[white]Score: {score} / {max_score}')


if __name__ == '__main__':
    main()
