


def optionsdict(f_key):
    # assert args is str, "make sure you pass a string arg to the options dictionary"
    return options[f_key]

options = {'0-test': ('test', 'expanding window'),
           '0-train': ('train', 'expanding window'),
           '1-test': ('test', 'time component'),
           '1-train': ('train', 'time component'),
           '2-test': ('test', 'spread component'),
           '2-train': ('train', 'spread component')
           }
