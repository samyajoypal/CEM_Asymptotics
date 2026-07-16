"""Assemble journal-template manuscripts from the modular paper sources."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.S)
    if match is None:
        raise RuntimeError(f"Could not extract {pattern!r}")
    return match.group(1).strip()


def main_manuscript() -> None:
    source = (PAPER / "main.tex").read_text()
    abstract = extract(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source)
    body = extract(
        r"(\\section\{Introduction\}.*?)(?=\\bibliographystyle)", source
    )
    preamble = r"""\documentclass[numsec,webpdf,modern,medium,namedate]{oup-authoring-template}

\onecolumn
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,graphicx}
\usepackage{microtype}
\graphicspath{{./}}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{assumption}{Assumption}
\newtheorem{definition}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newcommand{\ind}{\mathbf 1}

\begin{document}
\journaltitle{Journal of the Royal Statistical Society Series B: Statistical Methodology}
\DOI{}
\copyrightyear{2026}
\pubyear{2026}
\access{}
\appnotes{Original Article}
\firstpage{1}
\title[Inference for hard classification likelihoods]{Inference for Hard Classification Likelihoods: Moving-Boundary Curvature and Feasible Uncertainty Quantification}

\author[1,$\ast$]{Samyajoy Pal\ORCID{0000-0003-1339-0979}}
\author[1]{Rouven E. Haschka}
\authormark{Pal and Haschka}
\address[1]{\orgdiv{Chair of Data Analytics}, \orgname{Technical University of Kaiserslautern-Landau}, \orgaddress{\street{Gottlieb-Daimler-Stra\ss e 42}, \postcode{67663}, \state{Kaiserslautern}, \country{Germany}}}
\corresp[$\ast$]{Samyajoy Pal, Chair of Data Analytics, Technical University of Kaiserslautern-Landau, 67663 Kaiserslautern, Germany. \href{mailto:samyajoy.pal@rptu.de}{samyajoy.pal@rptu.de}}

"""
    output = (
        preamble
        + "\\abstract{" + abstract + "}\n"
        + r"\keywords{classification likelihood; decision boundary; finite mixture; M-estimation; nonsmooth inference; sandwich covariance}"
        + "\n\\maketitle\n\n"
        + body
        + "\n\\bibliographystyle{abbrvnat}\n\\bibliography{references}\n\\end{document}\n"
    )
    (PAPER / "jrssb_main.tex").write_text(output)


def supplement() -> None:
    output = r"""\documentclass[numsec,webpdf,modern,medium,namedate]{oup-authoring-template}
\onecolumn
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,longtable,array,graphicx,microtype}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{assumption}[theorem]{Assumption}
\newcommand{\ind}{\mathbf 1}
\begin{document}
\journaltitle{Journal of the Royal Statistical Society Series B: Statistical Methodology}
\DOI{}
\copyrightyear{2026}
\pubyear{2026}
\access{}
\appnotes{Supplementary Material}
\firstpage{1}
\title[Supplementary material]{Supplementary Material for ``Inference for Hard Classification Likelihoods: Moving-Boundary Curvature and Feasible Uncertainty Quantification''}
\author[1,$\ast$]{Samyajoy Pal\ORCID{0000-0003-1339-0979}}
\author[1]{Rouven E. Haschka}
\authormark{Pal and Haschka}
\address[1]{\orgdiv{Chair of Data Analytics}, \orgname{Technical University of Kaiserslautern-Landau}, \orgaddress{\street{Gottlieb-Daimler-Stra\ss e 42}, \postcode{67663}, \state{Kaiserslautern}, \country{Germany}}}
\corresp[$\ast$]{\href{mailto:samyajoy.pal@rptu.de}{samyajoy.pal@rptu.de}}
\abstract{Technical proofs, complete simulation summaries, bandwidth and failure diagnostics, real-data screening results, and parameter-level application results.}
\keywords{classification likelihood; supplementary material}
\maketitle
\renewcommand{\theequation}{S.\arabic{equation}}
\setcounter{equation}{0}
\input{appendix/technical_proofs}
\input{appendix/empirical_details}
\bibliographystyle{abbrvnat}
\bibliography{references}
\end{document}
"""
    (PAPER / "jrssb_supplement.tex").write_text(output)


if __name__ == "__main__":
    main_manuscript()
    supplement()
