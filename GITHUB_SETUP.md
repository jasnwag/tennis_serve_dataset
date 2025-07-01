# GitHub Repository Setup Guide

## Creating the Repository

### Option 1: Using GitHub CLI (Recommended)
1. Install GitHub CLI:
   ```bash
   # On macOS
   brew install gh
   
   # On Ubuntu/Debian
   sudo apt install gh
   
   # On Windows
   winget install GitHub.cli
   ```

2. Authenticate with GitHub:
   ```bash
   gh auth login
   ```

3. Create the repository:
   ```bash
   gh repo create tennis-analysis --public --description "Tennis match analysis project with serve analysis, gender classification, and scorebug detection" --source=. --remote=origin --push
   ```

### Option 2: Manual Setup
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right and select "New repository"
3. Repository name: `tennis-analysis`
4. Description: "Tennis match analysis project with serve analysis, gender classification, and scorebug detection"
5. Make it Public or Private (your choice)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

4. Connect your local repository to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/tennis-analysis.git
   git branch -M main
   git push -u origin main
   ```

## Repository Features

✅ **Already Set Up:**
- Comprehensive `.gitignore` file (excludes data/, cache files, etc.)
- Detailed `README.md` with project documentation
- `requirements.txt` with all dependencies
- `setup_data.py` script for data management
- Complete code structure with 55+ files

✅ **Data Management:**
- Data directory excluded from repository
- Setup script for creating data structure on other computers
- Transfer instructions for cross-computer data access

## Next Steps

1. **Create the GitHub repository** using one of the methods above
2. **Push your code** to GitHub
3. **Share the repository** with collaborators
4. **Set up data** on other computers using `python setup_data.py --all`

## Data Transfer Options

When setting up on another computer:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tennis-analysis.git
   cd tennis-analysis
   ```

2. **Set up data structure:**
   ```bash
   python setup_data.py --all
   ```

3. **Transfer data** using one of these methods:
   - **Cloud Storage**: Upload to Google Drive/Dropbox/OneDrive
   - **External Drive**: Copy data/ directory
   - **Network Transfer**: Use rsync or scp
   - **Git LFS**: For smaller datasets

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

```
tennis-analysis/
├── .gitignore              # Excludes data and cache files
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── setup_data.py          # Data setup script
├── code/                  # All source code
│   ├── notebooks/         # Jupyter notebooks
│   ├── openai/           # OpenAI API scripts
│   └── src/              # Main analysis code
└── data/                 # Data directory (not in repo)
    ├── initial/          # Processed data
    ├── scorebug/         # Scorebug data
    ├── USTA/            # USTA data
    └── visualizations/  # Generated plots
```

## Benefits of This Setup

- **Clean Repository**: Only code and documentation, no large data files
- **Easy Collaboration**: Others can clone and set up easily
- **Data Flexibility**: Multiple ways to transfer data between computers
- **Professional Structure**: Well-organized code with proper documentation
- **Scalable**: Easy to add new features and maintain 