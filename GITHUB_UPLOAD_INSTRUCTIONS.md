# GitHub Upload Instructions

This folder is a Git-ready public code release package for the revised manuscript.

Before publishing:

1. Review the repository contents once more.
2. Choose a public repository name, for example:
   `gc-myeloid-spatial-reproducibility`
3. Select a license with all authors' agreement.
   The current placeholder file is `LICENSE_PENDING_AUTHOR_CONFIRMATION.txt`.
4. Do not use account passwords for Git operations. Use a browser login, GitHub Desktop,
   GitHub CLI after authentication, or a personal access token.

Recommended GitHub web route:

1. Sign in to GitHub in a browser.
2. Create a new public repository.
3. Upload or push the contents of this folder.
4. After the repository is public, record the final URL and add it to the manuscript
   Code Availability statement.

Command-line route after creating an empty GitHub repository:

```bash
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

Replace `<owner>` and `<repo>` with the actual GitHub owner and repository name.
