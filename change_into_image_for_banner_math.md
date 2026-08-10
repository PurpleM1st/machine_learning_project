CrossEntropyLoss
       ↓
"What is the error?"

       ↓

loss.backward()
       ↓
"What is the gradient for each weight?"

       ↓

AdamW
       ↓
"How should I update the weights?"

       ↓

CosineAnnealingLR
       ↓
"How large should future updates be?"