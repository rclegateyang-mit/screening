# Census RDC Computing Environment

Reference for writing code that runs on Census FSRDC (Federal Statistical Research Data Center) servers.
No internet or AI tools are available inside the RDC.

Sources: FSRDC Researcher Handbook (Feb 2026), Thin Client User Guide (Feb 2018),
Disclosure Avoidance Methods Handbook v5.0 (Oct 2024), individual RDC site pages.

---

## Architecture

- **Thin client model**: Researchers sit at diskless terminals at physical RDC locations.
- Thin clients connect via encrypted lines to Census Bureau's **Virtual Desktop Infrastructure (VDI)**
  using **NoMachine Enterprise Client (NX)**.
- All data storage and processing on central servers at Census Computer Center in **Bowie, MD**.
- No Title 13 or Title 26 data is physically at any RDC site.
- **Secure Remote Access (SRA)** available for eligible researchers from a designated home workspace.

## Operating System

- **Red Hat Linux** with KDE desktop environment.
- Terminal: Konsole. File manager: Dolphin. Text editors: kate, kedit, kwrite, emacs.

## Server Cluster (IRE -- Integrated Research Environment)

- **Login nodes**: `hpc-login1.rm.census.gov`, `hpc-login2.rm.census.gov`
  (load balancer: `hpc-login.rm.census.gov`).
- **Compute nodes**: `scompute 1-9`, `hcompute 1-12`.
- **Analytical processes CANNOT run on login nodes** -- all jobs via PBS Pro.
- Default PBS allocation: **1 CPU, 5 GB memory**. Can request more.
- Shared resources -- all researchers share disk, RAM, and CPU.

## Job Scheduler: PBS Professional (PBS Pro)

All statistical jobs must be submitted through PBS Pro. **Do not run compute on login nodes.**

### Batch submission (preferred for production runs)

Wrapper scripts:
```bash
qsas program_name.sas &
qstata program_name.do &
qR program_name &
```

Custom batch script:
```bash
#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb
cd /projects/programs
stata-se -b do program_name.do
```

Python batch script:
```bash
#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
cd /projects/<project_id>/programs
python my_script.py
```

Submit: `qsub example.bash &`

### Interactive submission (for debugging only)

```bash
qsub -I -X                                    # default resources
qsub -I -X -l select=1:ncpus=1:mem=7gb       # custom resources
```

Interactive jobs **auto-terminate after 10-12 hours**. Type `exit` to end.

### Monitoring

```bash
qstat          # view job queue
qstat -a       # show all jobs
qstat -n       # show execution node
qstat -f       # full details
qstat -fan     # combined detailed view
pbsnodes -a    # node status and available resources
qdel <job_id>  # kill your own job
```

PBS output files land in `/projects/logs/`:
- `jobnumber.hpc-app1.rm.census.gov.OU` (stdout)
- `jobnumber.hpc-app1.rm.census.gov.ER` (stderr)

---

## Available Software

| Application | Notes |
|---|---|
| **Python (Anaconda)** | Standard Anaconda distribution |
| **R + RStudio** | |
| **SAS** (v9.4) | Primary supported; all Census data in SAS format |
| **Stata** (SE + MP) | Primary supported; only **5 Stata-MP licenses** |
| **MATLAB** | Limited shared licenses |
| **Mathematica** | |
| **Fortran** | Intel Composer compilers |
| **Perl** | |
| **Gauss** | |
| **Gurobi** | Optimization (+ related Anaconda modules) |
| **Tomlab / KNITRO / MADD** | Optimization |
| **QGIS / OpenGeoDa** | GIS / spatial |
| **SUDAAN** | SAS-callable survey analysis |
| **Stat/Transfer** | Data format conversion |
| **TeX/LaTeX** | |
| **OpenOffice** | Word processor, spreadsheet |
| **PBS Pro** | Job scheduler |

## Python Environment

### Activation (required on compute nodes)

Anaconda is NOT on `PATH` by default on compute nodes. **Must activate before any Python use:**

```bash
source /apps/anaconda/bin/activate py3cf    # conda-forge, most complete
```

Available environments (check with `conda env list`):
- `py3` -- standard Python 3
- `py3_spatial` -- adds mapping/GIS features
- `py3cf` -- conda-forge, most current Python, most complete packages (recommended)
- `base` -- minimal, may lack key packages

Note: `python` may not exist; use `python3` or activate an environment first.

### What's available

- **Anaconda distribution** is installed (ships with ~300 packages).
- Almost certainly includes: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `statsmodels`.
- Run `python --version` and `conda list` on first login to confirm versions.

### What's NOT available (must be requested)

- **JAX / jaxlib** -- not in standard Anaconda. High-risk dependency for this project.
- **mpi4py** -- not confirmed available. MPI runtime status unknown.
- **Any package not in the Anaconda distribution**.

### Package installation

- **No internet access** -- `pip install` and `conda install` from remote repos are impossible.
- Must request through RDCA (RDC Administrator) / Census IT.
- Include package requests in research proposal under "software/add-ons needed".
- Must justify why existing packages cannot do the work.
- For licensed software, research team pays for the license.
- **Request early** -- no documented timeline for package installation.

### Fallback strategy for this project

If JAX is unavailable, the estimation code needs a **scipy-only fallback**:
- `scipy.optimize.minimize(method='L-BFGS-B')` replaces JAX-based optimization.
- `numpy` for array operations (no autodiff -- need analytical or numerical gradients).
- Single-process estimation (no MPI) unless mpi4py is confirmed.
- `multiprocessing` module (stdlib) may work for parallelism within a single PBS node.

## Data Format

- **All Census data provided in SAS format** (`.sas7bdat`).
- Even non-SAS users need SAS to convert data. Use `Stat/Transfer` or `proc export` in SAS.
- Plan a SAS-to-CSV or SAS-to-pickle conversion step.

```sas
/* Example: export SAS to CSV */
proc export data=mylib.mydata
    outfile="/projects/<project_id>/data/mydata.csv"
    dbms=csv replace;
run;
```

Python can also read SAS directly (if pandas version supports it):
```python
import pandas as pd
df = pd.read_sas('/projects/<project_id>/data/mydata.sas7bdat')
```

---

## Directory Structure

### Source data
```
/data/economic/       # LBD, ASM, Census of Manufactures, etc.
/data/decennial/      # Decennial census
/data/demographic/    # ACS, SIPP, CPS
```

### Project directory
```
/projects/<project_id>/
    bin/              # project-specific scripts
    data/             # input/output files shared among project users
    disclosure/       # output for disclosure review
    etc/              # system-level configs
    logs/             # PBS job log files
    programs/         # scripts and programs
    transfer/         # data placed by Census data staff
    users/            # per-user directories (users/{username})
```

### Home directory
```
/home/<a-z>/<User_ID>/
```
For personal config only (`.bashrc`, `.bash_profile`). Project files go in `/projects/`.

### Shared resources
- IRE Code Library: `/data/support/researcher/codelib/`
- Shared Stata .ado files: `/apps/shared/stata/`
- SAS installation: `/apps/SAS/v9.4/`

### Internal documentation (accessible from inside RDC only)
- `http://rdcdoc.ces.census.gov`
- `http://rdcdoc.cods.census.gov`

---

## Resource Limits and "Good Citizen" Rules

- **Debug interactively on a SUBSET of data**, then run production in batch.
- Close Stata/MATLAB when not actively using them (they hold memory).
- Don't open more than one interactive Stata session.
- **Notify your administrator before running memory-intensive programs.**
- Census reserves the right to **kill processes** creating system instability.
- Interactive sessions timeout after **10-12 hours** -- cannot run overnight interactively.
- No formal storage quotas, but shared filesystem -- compress and delete unused files.
- Project space stays on IRE for **2 years after project completion**, then archived to tape.

---

## Security Rules

- **No internet, email, or web browsing** while in the RDC.
- **No laptops, tablets, USB drives, CD/DVD** in the lab.
- Cell phones allowed but: no calls, texts, email, browsing, streaming. Pre-downloaded music only.
- Badge in every time. Never let others tailgate.
- Lock workstation when stepping away.
- Shred handwritten notes before leaving (cannot be removed).
- All printers watermark: "Disclosure Prohibited -- Title 13 U.S.C. and Title 26 U.S.C."
- All user activities are logged and monitored.
- Password: 12+ chars, mixed case + number + special. Expires every 60 days.

---

## External Reference URLs

| Document | URL |
|---|---|
| FSRDC Main Page | https://www.census.gov/about/adrm/fsrdc.html |
| Researcher Handbook (PDF) | https://www2.census.gov/adrm/FSRDC/Resources/FSRDC-Researcher-Handbook.pdf |
| Thin Client User Guide (PDF) | https://www2.census.gov/adrm/FSRDC/Resources/FSRDC-Thin-Client-User-Guide.pdf |
| DA Methods Handbook v5.0 (PDF) | https://www2.census.gov/adrm/FSRDC/Resources/FSRDC-Disclosure-Avoidance-Methods-Handbook.pdf |
| DA Procedures Handbook (PDF) | https://www2.census.gov/adrm/FSRDC/Resources/FSRDC-Disclosure-Avoidance-Procedures-Handbook.pdf |
| Research Proposal Guidelines (PDF) | https://www2.census.gov/adrm/FSRDC/Apply_For_Access/Research_Proposal_Guidelines.pdf |
| Secure Remote Access | https://www.census.gov/about/adrm/fsrdc/about/secure-remote-access.html |
| Password Self-Service | https://pss.tco.census.gov |
