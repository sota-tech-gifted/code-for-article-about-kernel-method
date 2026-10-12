"""
Standalone evaluation script: load a trained model and measure accuracy on target domains.
"""
import argparse, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_dg import ResNet50Linear

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--source', default='photo')
    parser.add_argument('--targets', default='art_painting,sketch,cartoon')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    src_dataset = datasets.ImageFolder(f'{args.data_root}/{args.source}', transform=test_transform)
    num_classes = len(src_dataset.classes)

    model = ResNet50Linear(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt))
    model.cuda().eval()

    target_domains = [d.strip() for d in args.targets.split(',')]
    accs = []
    for dom in target_domains:
        ds = datasets.ImageFolder(f'{args.data_root}/{dom}', transform=test_transform)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in dl:
                imgs, labels = imgs.cuda(), labels.cuda()
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        acc = 100.0 * correct / total
        accs.append(acc)
        print(f'{dom:>15s}: {acc:.2f}%')
    print(f'{"Average":>15s}: {sum(accs)/len(accs):.2f}%')

if __name__ == '__main__':
    main()
