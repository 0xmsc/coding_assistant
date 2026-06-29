---
name: pdf-text-extraction
description: Extract readable text from PDF files with pdftotext. Use when a task involves uploaded PDF attachments, local .pdf files, or the agent needs to read, summarize, search, convert, or answer questions from a PDF.
---

# PDF Text Extraction

Use `pdftotext` in the worker shell to convert text-based PDFs into workspace text files, then inspect the extracted text with `rg`, `sed`, Python, or normal file reads.

## Workflow

1. Create an output directory in the writable workspace:

```bash
mkdir -p extracted
```

2. Extract text while preserving useful layout:

```bash
pdftotext -layout -enc UTF-8 /attachments/file.pdf extracted/file.txt
```

3. Check that extraction produced meaningful text before relying on it:

```bash
wc -c extracted/file.txt
sed -n '1,80p' extracted/file.txt
```

4. Search or read the extracted file and cite the PDF content from the extracted text, not from the filename.

For multiple PDFs, use distinct output filenames, preferably including the attachment id or PDF stem.

If extraction output is empty or mostly page breaks, the PDF is probably scanned or image-only. State that OCR is required; do not infer the contents.
