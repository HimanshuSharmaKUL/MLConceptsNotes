---
sticker: emoji//26d3-fe0f
---
[[2 - Tags/NLP|NLP]] 
Memory Types:
- ConversationBufferMemory
	- Allows to store messages and extracts messages in a variable
- ConversationBufferWindowMemory
- ConversationTokenBufferMemory
- ConversationSummaryMemory
- Vector Data Memory
- Entity Memory
- COmbination of memorys - conversation + entity etc

Chains
- LLM Chain
- Sequential Chains
	- SimpleSequentialChain
		- ![[Pasted image 20250706153129.png]]
	- SequentialChain
		- ![[Pasted image 20250706153103.png]]
- Router Chain
	- ![[Pasted image 20250706153001.png]]


Stuff Method:
![[Pasted image 20250707150408.png]]
Easy and simple method, but will get computationally and money wise costly as we use bigger documents. We can use other methods then: 1st most common: Stuff Method. 2nd most common: Map_reduce (used for summarisation)

![[Pasted image 20250707150728.png]]

Langchain Evaluation
![[Pasted image 20250707155000.png]]

# Chat with your Data - Langchain
Components
Prompts
• Prompt Templates
• Output Parsers: 5+ implementations
- Retry/fixing logic
• Example Selectors: 5+ implementations
Models
• LLM's: 20+ integrations
• Chat Models
• Text Embedding Models: 10+ integrations
Indexes
Document Loaders: 50+ implementations
Text Splitters: 10+ implementations
Vector stores: 10+ integrations
Retrievers: 5+ integrations/implementations
•
Chains
o Can be used as building blocks for other chains
o More application specific chains: 20+ different types
Agents
o Agent Types: 5+ types
Algorithms for getting LLMs to use tools
o Agent Toolkits: 10+ implementations
Agents armed with specific tools for a
specific application

![[Pasted image 20250715235756.png]]

## Loaders
Loaders deal with the specifics of accessing and converting data
- Accessing
• Web Sites
• Data Bases
• YouTube
• arXiv
- Data Types
• PDF
• HTML
• JSON
• Word, PowerPoint.„

Returns a list of `Document` objects:
[
Document(page_content='MachineLearning-Lecture01 \nlnstructor (Andrew Ng): Okay.
Good moming. Welcome to CS229....',
metadata = {'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf, 'page': 0})
...
Document(page_content='[End of Audio] \nDuration: 69 minutes`,
metadata = {'source' : 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 21})
]

![[Pasted image 20250716004508.png]]

