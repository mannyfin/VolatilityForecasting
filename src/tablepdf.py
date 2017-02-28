import pandas as pd
import numpy as np
import subprocess

def tablepdfmaker(df):
    filename = 'output.tex'
    pdffile = 'out.pdf'
    outname = 'out.png'

    import pandas as pd
    df = pd.DataFrame.from_dict(df)

    with open("table.tex", "w") as f:
        f.write("\\begin{tabular}{" + " | ".join(["c"] * len(df.columns)) + "}\n")
        for i, row in df.iterrows():
            f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
        f.write("\\end{tabular}")

