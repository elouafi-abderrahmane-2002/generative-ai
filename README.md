# 📄 Gemini + LangChain — Document Summarization Pipeline

Investment firms, VCs and research teams deal with a constant flood of documents :
pitch decks, market reports, financial statements, news articles, due diligence memos.
Reading everything manually is impossible. This pipeline uses Gemini 2.0 Flash via
LangChain to automatically summarize any document — PDF, web page or raw text —
into structured, actionable insights.

---

## Three summarization strategies

```
  Document
      │
      ├── SHORT (<4000 tokens)
      │       │
      │       ▼
      │   STUFFING — send the whole document in one prompt
      │   Fast, simple, works well for short reports
      │
      ├── MEDIUM / LONG (>4000 tokens)
      │       │
      │       ▼
      │   MAP-REDUCE — split into chunks, summarize in parallel
      │   then combine all summaries into a final synthesis
      │   Best for : annual reports, long research papers
      │
      └── ITERATIVE
              │
              ▼
          REFINE — summarize chunk 1, then refine with chunk 2,
          then refine again with chunk 3...
          Best for : documents with strong narrative continuity
```

---

## Architecture

```
  Input (PDF / URL / text)
          │
          ▼
  DocumentLoader
  ├── PyPDFLoader      ← PDF files
  ├── WebBaseLoader    ← web articles, reports
  └── TextLoader       ← raw text, markdown
          │
          ▼
  TextSplitter
  RecursiveCharacterTextSplitter(
      chunk_size=2000,
      chunk_overlap=200   ← overlap keeps context across chunks
  )
          │
          ▼
  SummarizationChain (Gemini 2.0 Flash)
  ├── load_summarize_chain("stuff")
  ├── load_summarize_chain("map_reduce")
  └── load_summarize_chain("refine")
          │
          ▼
  Structured Output
  {
    "executive_summary": "...",
    "key_points":        ["...", "...", "..."],
    "action_items":      ["...", "..."],
    "risk_flags":        ["..."]
  }
```

---

## Core implementation

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.prompts import PromptTemplate

# Initialize Gemini 2.0 Flash
llm = ChatGoogleGenerativeAI(
    model       = "gemini-2.0-flash",
    google_api_key = os.environ["GOOGLE_API_KEY"],
    temperature = 0.2   # low temp for factual summarization
)

# Custom prompts — tailored for investment research
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
You are an investment research analyst. Analyze this section and extract:
- Key business insights
- Financial figures or metrics mentioned
- Risks or red flags
- Market opportunities

Section:
{text}

Structured analysis:"""
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
You are a senior investment analyst. Based on these section analyses,
write a concise executive summary suitable for an investment committee.

Format:
**EXECUTIVE SUMMARY** (2-3 sentences)
**KEY INSIGHTS** (bullet points)
**FINANCIAL HIGHLIGHTS** (if any)
**RISKS TO MONITOR** (bullet points)
**RECOMMENDATION** (one sentence)

Analyses:
{text}

Investment memo:"""
)

class DocumentSummarizer:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size    = 2000,
            chunk_overlap = 200,
        )

    def summarize_pdf(self, pdf_path: str, strategy: str = "auto") -> dict:
        """
        Summarize a PDF document using the optimal strategy.
        strategy: "auto" | "stuff" | "map_reduce" | "refine"
        """
        loader = PyPDFLoader(pdf_path)
        docs   = loader.load()
        return self._summarize(docs, strategy)

    def summarize_url(self, url: str, strategy: str = "auto") -> dict:
        """Summarize a web article or online report."""
        loader = WebBaseLoader(url)
        docs   = loader.load()
        return self._summarize(docs, strategy)

    def _summarize(self, docs: list, strategy: str) -> dict:
        # Auto-select strategy based on document length
        total_tokens = sum(len(d.page_content.split()) for d in docs)

        if strategy == "auto":
            strategy = "stuff" if total_tokens < 3000 else "map_reduce"

        if strategy == "stuff":
            chain = load_summarize_chain(llm, chain_type="stuff")
        elif strategy == "map_reduce":
            chain = load_summarize_chain(
                llm,
                chain_type      = "map_reduce",
                map_prompt      = MAP_PROMPT,
                combine_prompt  = COMBINE_PROMPT,
                verbose         = False
            )
        elif strategy == "refine":
            chain = load_summarize_chain(llm, chain_type="refine")

        # Split and summarize
        split_docs = self.splitter.split_documents(docs)
        result     = chain.invoke({"input_documents": split_docs})

        return {
            "summary":        result["output_text"],
            "strategy_used":  strategy,
            "pages_processed": len(docs),
            "chunks_processed": len(split_docs)
        }


# Usage example
summarizer = DocumentSummarizer()

# Summarize an investment report
result = summarizer.summarize_pdf("q3_africa_report.pdf")
print(result["summary"])

# Summarize a web article
result = summarizer.summarize_url("https://techcrunch.com/some-article")
print(result["summary"])
```

---

## Batch processing — summarize an entire folder

```python
from pathlib import Path
import json

def batch_summarize(input_dir: str, output_file: str):
    """
    Summarize all PDFs in a directory.
    Useful for processing a due diligence package at once.
    """
    summarizer = DocumentSummarizer()
    results    = {}

    for pdf in Path(input_dir).glob("*.pdf"):
        print(f"Processing : {pdf.name}...")
        try:
            results[pdf.name] = summarizer.summarize_pdf(str(pdf))
        except Exception as e:
            results[pdf.name] = {"error": str(e)}

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {len(results)} documents summarized → {output_file}")

# Process an entire due diligence folder
batch_summarize("due_diligence/", "summaries.json")
```

---

## What I learned

The **chunk overlap** parameter matters more than expected. Without overlap,
sentences cut at chunk boundaries lose their context — the summarizer misses
insights that span two chunks. With 200-token overlap, the model sees the
end of one chunk and the beginning of the next, preserving continuity.

For investment documents specifically, **MapReduce with custom prompts**
consistently outperforms the default chain. The default prompt is too generic —
asking explicitly for financial figures, risks and opportunities produces
summaries that are immediately usable in an investment memo without editing.

---

*Project built as part of Engineering degree — ENSET Mohammedia*
*By **Abderrahmane Elouafi** · [LinkedIn](https://www.linkedin.com/in/abderrahmane-elouafi-43226736b/) · [Portfolio](https://my-first-porfolio-six.vercel.app/)*
