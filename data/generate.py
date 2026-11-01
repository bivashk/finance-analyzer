"""Generate transparent synthetic fixtures; never use these scores as real-world claims."""
import csv
from pathlib import Path
from backend.analytics import CATEGORIES

TERMS = ['metro transport commute', 'restaurant cafe food', 'clothing shopping apparel',
 'cinema movie entertainment', 'grocery supermarket vegetables', 'electricity broadband utility',
 'salary payroll income', 'hotel resort booking', 'laptop electronics computer', 'petrol diesel fuel',
 'pharmacy clinic medicine', 'gym yoga fitness', 'rent landlord apartment', 'school tuition course',
 'account fee charge', 'miscellaneous unknown transfer']

if __name__ == '__main__':
    path = Path(__file__).with_name('transactions_synthetic.csv')
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['date','description','amount','category','merchant','split'])
        writer.writeheader()
        for category, terms in zip(CATEGORIES, TERMS):
            for merchant in range(8):
                for variant in range(6):
                    identity = f'merchant{CATEGORIES.index(category):02d}{merchant:02d}'
                    writer.writerow({'date':f'2026-{variant+1:02d}-{merchant+1:02d}',
                     'description': f'{identity} {terms} {"payment" if variant%2 else "purchase"} ref {variant}',
                     'amount': 20000 if category=='Income' else -(100+merchant*50+variant*10),
                     'category':category,'merchant':identity,'split':'test' if merchant>=6 else 'train'})
    print(f'Wrote {path}')
