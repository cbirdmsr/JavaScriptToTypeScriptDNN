import re

regex = re.compile(r"^[^\d\W]\w*$", re.UNICODE)
keywords = ["async", "await", "break", "continue", "class", "extends", "constructor", "super", "extends", "const", "let", "var", "debugger", "delete", "do", "while", "export", "import", "for", "each", "in", "of", "function", "return", "get", "set", "if", "else", "instanceof", "typeof", "null", "undefined", "switch", "case", "default", "this", "true", "false", "try", "catch", "finally", "void", "yield", "any", "boolean", "null", "never", "number", "string", "symbol", "undefined", "void", "as", "is", "enum", "type", "interface", "abstract", "implements", "static", "readonly", "private", "protected", "public", "declare", "module", "namespace", "require", "from", "of", "package"]

with open("results-4-5.csv", "r") as f:
    hits = 0
    misses = 0
    neither = 0
    mappings = {}
    trues = {}
    obvs = {}
    inconsistencies = set()
    inconsistenciesB = set()
    for line in f:
        parts = line.split(",")
        if parts[0][0] == "\"" and parts[0][len(parts[0])-1] == "\"":
            parts[0] = parts[0][1:len(parts[0])-1]
        if parts[0] == "<s>":
            hits += len(mappings) - len(inconsistencies) - len(obvs)
            misses += len(inconsistencies)
            neither += len(inconsistenciesB)
            mappings = {}
            trues = {}
            obvs = {}
            inconsistencies = set()
            inconsistenciesB = set()
        elif regex.match(parts[0]) and parts[0] not in keywords:
            if parts[0] not in mappings:
                mappings[parts[0]] = parts[2]
                trues[parts[0]] = parts[1]
                obvs[parts[0]] = parts[1]
            else:
                if parts[0] in obvs:
                    del obvs[parts[0]]
                if mappings[parts[0]] != parts[2]:
                    if trues[parts[0]] != parts[1]:
                        inconsistenciesB.add(parts[0])
                    else:
                        inconsistencies.add(parts[0])
    print("%d, %d, %d" % (hits, misses, neither))
    print("%.3f%%, %.3f%%" % (100*misses/(hits+misses), 100*neither/(hits+misses+neither)))

