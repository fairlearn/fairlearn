import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights } from "office-ui-fabric-react";

export interface IModelComparisionChartStyles {
    frame: IStyle;
    spinner: IStyle;
    header: IStyle;
    headerTitle: IStyle;
    editButton: IStyle;
    main: IStyle;
    mainRight: IStyle;
    rightTitle: IStyle;
    rightText: IStyle;
    insights: IStyle;
    insightsText: IStyle;
    chart: IStyle;
    textSection: IStyle;
    radio: IStyle;
}

export const ModelComparisionChartStyles: () => IProcessedStyleSet<IModelComparisionChartStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IModelComparisionChartStyles>({
        frame: {
            flex: 1,
            display: "flex",
            flexDirection: "column"
        },
        spinner: {
            margin: "auto",
            padding: "40px"
        },
        header: {
            //backgroundColor: "#EBEBEB",
            backgroundColor: theme.semanticColors.bodyFrameBackground, 
            padding: "0 90px",
            height: "90px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between",
            alignItems: "center"
        },
        headerTitle: {
            //color: "#333333",
            color: theme.semanticColors.bodyText,
            //fontSize: "32px",
            //lineHeight: "39px",
            //fontWeight: "100"
            fontWeight: FontWeights.light
        },
        editButton: {
            //color: "#333333",
            color: theme.semanticColors.buttonText,
            //fontSize: "12px",
            //lineHeight: "20px",
            //fontWeight: "400"
        },
        main: {
            height: "100%",
            flex: 1,
            display: "inline-flex",
            flexDirection: "row",
            backgroundColor: theme.semanticColors.bodyBackground
        },
        mainRight: {
            padding: "30px 0 0 35px",
            width: "300px"
        },
        rightTitle: {
            //color: "#333333",
            color: theme.semanticColors.bodyText,
            //fontSize: "15px",
            //lineHeight: "16px",
            //fontWeight: "500",
            paddingBottom: "18px",
            borderBottom: "1px solid",
            //borderColor:  "#CCCCCC"
            borderColor: theme.semanticColors.bodyDivider
        },
        rightText: {
            padding: "16px 15px 30px 0",
            //color: "#333333",
            color: theme.semanticColors.bodySubtext,
            //fontSize: "15px",
            //lineHeight: "18px",
            //fontWeight: "400",
            borderBottom: "0.5px dashed",
            //borderColor:  "#CCCCCC"
            borderColor: theme.semanticColors.bodyDivider
        },
        insights: {
            textTransform: "uppercase",
            //color: "#333333",
            color: theme.semanticColors.bodySubtext,
            //fontSize: "15px",
            //lineHeight: "16px",
            //fontWeight: "500",
            padding: "18px 0",
        },
        insightsText: {
            //color: "#333333",
            //color: theme.semanticColors.bodyText,
            //fontSize: "15px",
            //lineHeight: "16px",
            //fontWeight: "400",
            paddingBottom: "18px",
            paddingRight: "15px",
            borderBottom: "1px solid",
            //borderColor:  "#CCCCCC",
            borderColor: theme.semanticColors.bodyDivider
        },
        chart: {
            padding: "60px 0 0 0",
            flex: 1
        },
        textSection: {
            //color: "#333333"
            color: theme.semanticColors.bodySubtext,
            paddingBottom: "5px"
        },
        radio: {
            paddingBottom: "30px",
            paddingLeft: "75px",
            color: theme.semanticColors.buttonText
        }
    });
};