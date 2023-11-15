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
    network_types = [('Fully Connected Networks', 'fc'), ('Convolutional Networks', 'conv')]
    scores_by_type = {
        'Fully Connected Networks': {
            'max': 0,
            'score': 0
        },
        'Convolutional Networks': {
            'max': 0,
            'score': 0
        }
    }

    for network_type in network_types:
        print('=' * 50)
        print(f'[bold white]{network_type[0]}')
        print('=' * 50)
        for i, net_name in enumerate([net_name for net_name in gt.keys() if net_name.startswith(network_type[1])]):
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
                try:
                    verified = analyze(net, image, eps, true_label)
                except RuntimeError:
                    verified = False
                dt = perf_counter() - start

                print(f'  [white]{i+1}.{j+1}  {net_name}: {spec}')
                test_score = 0
                if verified and not gt_verified:
                    test_score = -2
                    print(f'    verified      [red]wrong    [white]([red]{test_score}[white])  ({dt:.3f}s)')
                elif verified and gt_verified:
                    test_score = 1
                    scores_by_type[network_type[0]]['max'] += 1
                    print(f'    verified      [green]correct  [white]([green]+{test_score}[white])  ({dt:.3f}s)')
                elif not verified and gt_verified:
                    scores_by_type[network_type[0]]['max'] += 1
                    print(f'    not verified  [yellow]wrong    [white]([yellow]+{test_score}[white])  ({dt:.3f}s)')
                else:
                    print(f'    not verified  [green]correct  [white]([green]+{test_score}[white])  ({dt:.3f}s)')
                scores_by_type[network_type[0]]['score'] += test_score
                print('[white]--------------------------------------------------')

        print(f'[bold white]  Score for {network_type[0]}: {scores_by_type[network_type[0]]["score"]} / {scores_by_type[network_type[0]]["max"]}')
        print('[white]--------------------------------------------------')
        score += scores_by_type[network_type[0]]['score']
        max_score += scores_by_type[network_type[0]]['max']

    print()
    print('=' * 50)
    for network_type in scores_by_type:
        print(f'[white]{network_type}: {scores_by_type[network_type]["score"]} / {scores_by_type[network_type]["max"]}')
    print(f'\n[white bold]Total Score: {score} / {max_score}')
    print('=' * 50)


if __name__ == '__main__':
    main()
