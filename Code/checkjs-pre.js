"use strict";
const ts = require("typescript");
const fs = require("fs");
const path = require("path");
function print(x) { console.log(x); }
var removableLexicalKinds = [
    ts.SyntaxKind.EndOfFileToken,
    ts.SyntaxKind.NewLineTrivia,
    ts.SyntaxKind.WhitespaceTrivia
];
var templateKinds = [
    ts.SyntaxKind.TemplateHead,
    ts.SyntaxKind.TemplateMiddle,
    ts.SyntaxKind.TemplateSpan,
    ts.SyntaxKind.TemplateTail,
    ts.SyntaxKind.TemplateExpression,
    ts.SyntaxKind.TaggedTemplateExpression,
    ts.SyntaxKind.FirstTemplateToken,
    ts.SyntaxKind.LastTemplateToken,
    ts.SyntaxKind.TemplateMiddle
];
let root = "C:/Users/t-vihell/Documents/TypeScriptStudy/Clones";
let rootCopy = "C:/Users/t-vihell/Documents/TypeScriptStudy/Clones-cleaned";
let outputDir = "C:/Users/t-vihell/Documents/TypeScriptStudy/outputs-checkjs-pre/";
try {
    fs.mkdirSync(outputDir);
}
catch (err) {
}
const ANY_THRESHOLD = 0.2;
fs.readdirSync(root).forEach(org => fs.readdirSync(root + "/" + org).forEach(project => traverseProject(org, project)));
function traverseProject(org, project) {
    let dir = root + "/" + org + "/" + project;
    let outFile = dir.substr(root.length + 1) + ".json";
    outFile = outFile.replace(/\//g, "__");
    outFile = outputDir + outFile;
    if (fs.existsSync(outFile))
        return;
    let projectTokens = traverse(dir);
    fs.writeFileSync(outFile, projectTokens.join("\n"), 'utf-8');
}
function traverse(dir) {
    var children = fs.readdirSync(dir);
    let tokens = [];
    if (children.find(value => value == "tsconfig.json")) {
        print("Config in: " + dir);
        tokens = tokens.concat(extractAlignedSequences(dir));
    }
    else {
        children.forEach(function (file) {
            let fullPath = dir + "/" + file;
            if (fs.statSync(fullPath).isDirectory()) {
                if (fullPath.indexOf("DefinitelyTyped") < 0) {
                    tokens = tokens.concat(traverse(fullPath));
                }
                else {
                    print("Skipping: " + fullPath);
                }
            }
        });
    }
    return tokens;
}
function extractAlignedSequences(inputDirectory) {
    const keywords = ["async", "await", "break", "continue", "class", "extends", "constructor", "super", "extends", "const", "let", "var", "debugger", "delete", "do", "while", "export", "import", "for", "each", "in", "of", "function", "return", "get", "set", "if", "else", "instanceof", "typeof", "null", "undefined", "switch", "case", "default", "this", "true", "false", "try", "catch", "finally", "void", "yield", "any", "boolean", "null", "never", "number", "string", "symbol", "undefined", "void", "as", "is", "enum", "type", "interface", "abstract", "implements", "static", "readonly", "private", "protected", "public", "declare", "module", "namespace", "require", "from", "of", "package"];
    let files = [];
    walkSync(inputDirectory, files, '.ts');
    let program = ts.createProgram(files, { target: ts.ScriptTarget.Latest, module: ts.ModuleKind.CommonJS });
    let checker = program.getTypeChecker();
    let allFileTokens = [];
    for (const sourceFile of program.getSourceFiles()) {
        let filename = sourceFile.getSourceFile().fileName;
        if (filename.endsWith('.d.ts'))
            continue;
        try {
            let relativePath = path.relative(inputDirectory, filename);
            if (relativePath.startsWith(".."))
                continue;
            let memS = [];
            let memT = [];
            extractTokens(sourceFile, checker, memS, memT);
            if (memS.length != memT.length)
                console.log(memS.length + ", " + memT.length);
            fs.writeFileSync(sourceFile.fileName.replace(root, rootCopy), memS.filter(val => val.length > 0).join(" "), 'utf-8');
            // Produce content and double-test for inconsistencies
            for (var ix in memS) {
                if (memS[ix].match("\".*\""))
                    memS[ix] = "\"s\"";
                else if (memS[ix].match("\'.*\'"))
                    memS[ix] = "\'s\'";
                else if (memS[ix].match("/.*/"))
                    memS[ix] = "/r/";
                else if (memS[ix].match("([0-9].*|\.[0-9].*)"))
                    memS[ix] = "0";
                memS[ix] = memS[ix].replace(/\\s/, "");
            }
            var content = memS.filter(val => val.length > 0).join(" ") + "\t" + memT.filter(val => val.length > 0).join(" ");
            var pretend = content.split("\t");
            var left = pretend[0].split(" ");
            var right = pretend[1].split(" ");
            if (left.length != right.length)
                console.log(left.length + ", " + right.length);
            allFileTokens.push(content);
        }
        catch (e) {
            console.log(e);
            console.log("Error parsing file " + filename);
        }
    }
    return allFileTokens;
}
function extractTokens(tree, checker, memS, memT) {
    var justPopped = false;
    for (var i in tree.getChildren()) {
        var ix = parseInt(i);
        var child = tree.getChildren()[ix];
        if (removableLexicalKinds.indexOf(child.kind) != -1 ||
            ts.SyntaxKind[child.kind].indexOf("JSDoc") != -1) {
            continue;
        }
        else if (templateKinds.indexOf(child.kind) != -1) {
            memS.push("`template`");
            memT.push("O");
            continue;
        }
        if (child.getChildCount() == 0) {
            var source = child.getText();
            var target = "O";
            switch (child.kind) {
                case ts.SyntaxKind.Identifier:
                    try {
                        let symbol = checker.getSymbolAtLocation(child);
                        let type = checker.typeToString(checker.getTypeOfSymbolAtLocation(symbol, child));
                        if (checker.isUnknownSymbol(symbol) || type.startsWith("typeof"))
                            target = "$any$";
                        else if (type.startsWith("\""))
                            target = "O";
                        else if (type.match("[0-9]+"))
                            target = "O";
                        else
                            target = '$' + type + '$';
                        break;
                    }
                    catch (e) { }
                    break;
                case ts.SyntaxKind.NumericLiteral:
                    target = "O";
                    break;
                case ts.SyntaxKind.StringLiteral:
                    target = "O";
                    break;
                case ts.SyntaxKind.RegularExpressionLiteral:
                    target = "O";
                    break;
            }
            target = target.trim();
            if (target.match(".+ => .+")) {
                target = "$" + target.substring(target.lastIndexOf(" => ") + 4);
            }
            if (target.match("\\s")) {
                target = "$complex$";
            }
            if (source.length == 0 || target.length == 0) {
                continue;
            }
            if (target != "O") {
                var parentKind = ts.SyntaxKind[tree.kind];
                if (parentKind.toLowerCase().indexOf("template") >= 0)
                    target = "O";
            }
            if (memS.length > 0 && memS[memS.length - 1] == ":" && Boolean(source.match("[a-zA-Z$_][a-zA-Z\$\_\[\]]*"))) {
                var k = tree.kind;
                var valid = k == ts.SyntaxKind.FunctionDeclaration || k == ts.SyntaxKind.MethodDeclaration || k == ts.SyntaxKind.Parameter || k == ts.SyntaxKind.VariableDeclaration;
                if (!valid && tree.kind == ts.SyntaxKind.TypeReference) {
                    k = tree.parent.kind;
                    valid = k == ts.SyntaxKind.FunctionDeclaration || k == ts.SyntaxKind.MethodDeclaration || k == ts.SyntaxKind.Parameter || k == ts.SyntaxKind.VariableDeclaration;
                }
                if (valid) {
                    memS.pop();
                    memT.pop();
                    justPopped = true;
                    continue;
                }
            }
            else if (justPopped) {
                if (source == "[" || source == "]")
                    continue;
                else
                    justPopped = false;
            }
            memS.push(source);
            memT.push(target);
        }
        else {
            extractTokens(child, checker, memS, memT);
        }
    }
}
function walkSync(dir, filelist, suffix) {
    var fs = fs || require('fs'), files = fs.readdirSync(dir);
    filelist = filelist || [];
    files.forEach(function (file) {
        let fullPath = path.join(dir, file);
        try {
            if (fs.statSync(fullPath).isDirectory()) {
                filelist = walkSync(dir + '/' + file, filelist, suffix);
            }
            else if (file.endsWith('.ts')) {
                filelist.push(fullPath);
            }
        }
        catch (e) {
            console.error("Error processing " + file);
        }
    });
    return filelist;
}
;
//# sourceMappingURL=checkjs-pre.js.map