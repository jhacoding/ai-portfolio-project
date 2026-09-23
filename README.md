AI Portfolio — Live Interview Transcript

An AI-powered interview chatbot embedded in my personal portfolio site. Instead of a static resume, visitors can ask open-ended questions — 
"What's your most recent role?", "What are your core skills?" — and get answers generated live by an LLM, speaking in first person as me. 
Every answer is grounded strictly in my actual resume, so it can't invent experience that isn't real.

The UI is styled as a live interview transcript: questions and answers stream in as timestamped entries in a running log, rather than a static FAQ page.

Live demo: https://raunak-ai-portfolio.vercel.app/

How it works
1) The frontend (static HTML/CSS/JS) is served from Vercel.
2) A visitor types a question and hits Ask — the frontend POSTs it to the FastAPI backend's /chat route on Render.
3) On its first request, the backend reads resume.pdf, sends the extracted text to the Groq LLM with a strict JSON schema, and caches the resulting structured
   Resume object in memory for the life of the server instance — so parsing only happens once, not on every question.
5) The cached resume data is embedded into a system prompt that instructs the model to answer in first person, as the candidate, using only resume facts.
6) The backend calls Groq with stream=True and yields each chunk as it arrives, wrapped in a FastAPI StreamingResponse.
7) The frontend reads the response body with response.body.getReader() and appends each decoded chunk to the DOM as it arrives — producing a live, token-by-token
   "typing" effect instead of one blocking wait.

Project structure
ai-portfolio-project/
├── Backend/
│   └── api/
│       ├── index.py         # FastAPI app, /chat route
│       ├── resume.pdf       # source resume, parsed once per instance
│       └── requirements.txt
└── Frontend/
    ├── index.html
    ├── style.css
    └── script.js
