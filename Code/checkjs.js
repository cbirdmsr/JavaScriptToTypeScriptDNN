"use strict";
const ts = require("typescript");
const fs = require("fs");
var removableLexicalKinds = [
    ts.SyntaxKind.EndOfFileToken,
    ts.SyntaxKind.NewLineTrivia,
    ts.SyntaxKind.WhitespaceTrivia,
];
function print(x) {
    console.log(x);
}
let infile = process.argv[2];
let outfile = infile.replace(".js", "") + "-out.txt"
extractTypes(infile);
function extractTypes(infile) {
    try {
        let program = ts.createProgram([infile], { target: ts.ScriptTarget.Latest, module: ts.ModuleKind.CommonJS, allowJs: true, checkJs: true });
    	let checker = program.getTypeChecker();
    	let sourceFile = program.getSourceFiles()[program.getSourceFiles().length - 1];
    	let data = extractTokens(sourceFile, checker)
        let content = ""
        for (var i = 0; i < data[0].length; i++) content += data[0][i] + String.fromCharCode(31) + data[1][i] + '\n'
        fs.writeFileSync(outfile, content, 'utf-8');
    }
    catch (e) {
        console.log("Error parsing file " + infile);
    }
}
function extractTokens(tree, checker) {
    var texts = [[], []];
    for (var i in tree.getChildren()) {
        var ix = parseInt(i);
        var child = tree.getChildren()[ix];
        if (removableLexicalKinds.indexOf(child.kind) != -1 ||
            ts.SyntaxKind[child.kind].indexOf("JSDoc") != -1) {
            continue;
        }
        if (child.getChildCount() == 0) {
            var source = child.getText()
            var target = "O"
            switch (child.kind) {
                case ts.SyntaxKind.Identifier:
                    try {
                        let symbol = checker.getSymbolAtLocation(child);
                        let type = checker.typeToString(checker.getTypeOfSymbolAtLocation(symbol, child));
                        if (checker.isUnknownSymbol(symbol) || type.startsWith("typeof")) target = "$any$";
                        else target = '$' + type + '$';
		    } catch (e) { }
		    break;
		case ts.SyntaxKind.NumericLiteral:
		    target = "O"
		    source = 'numlit'
		    break;
		case ts.SyntaxKind.StringLiteral:
		    target = "O"
		    source = 'stringlit'
		    break;
		case ts.SyntaxKind.RegularExpressionLiteral:
		    target = "O"
		    source = 'regexlit'
		    break;
		case ts.SyntaxKind.TemplateHead:
		case ts.SyntaxKind.TemplateMiddle:
		case ts.SyntaxKind.TemplateSpan:
		case ts.SyntaxKind.TemplateTail:
		case ts.SyntaxKind.TemplateExpression:
		case ts.SyntaxKind.TaggedTemplateExpression:
		case ts.SyntaxKind.FirstTemplateToken:
		case ts.SyntaxKind.LastTemplateToken:
		case ts.SyntaxKind.TemplateMiddle:
		    target = 'O'
		    source = 'template'
		    break;
		}
		source = source.trim()
		target = target.trim()
		if (target.match(".+ => .+")) {
		    target = "$" + target.substring(target.lastIndexOf(" => ") + 4)
		}
		if (target.match("\\s")) {
		    target = "$complex$"
		}
		if (source.match("\\s")) {
		    console.log("ERROR: " + source)
		    continue
		}
		if (source.length == 0 || target.length == 0) {
		    continue
		}
		if (target != "O") {
		    var parentKind = ts.SyntaxKind[tree.kind]
		    if (parentKind.toLowerCase().indexOf("template") >= 0)
		        target = "O"
		}
		texts[0].push(source)	
		texts[1].push(target)
        }
        else {
            var childTexts = extractTokens(child, checker);
            texts[0] = texts[0].concat(childTexts[0]);
            texts[1] = texts[1].concat(childTexts[1]);
        }
    }
    return texts;
}
//# sourceMappingURL=checkjs.js.map