import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, ThemeSettingName } from "office-ui-fabric-react";

export interface IModelComparisionChartStyles {
    frame: IStyle;
    spinner: IStyle;
    header: IStyle;
    headerTitle: IStyle;
    headerOptions: IStyle;
    dropDown: IStyle;
    doneButton: IStyle;
    infoButton: IStyle;
    modalContentIntro: IStyle;
    modalContentHelp: IStyle;
    editButton: IStyle;
    howTo: IStyle;
    main: IStyle;
    mainLeft: IStyle;
    mainRight: IStyle;
    rightTitle: IStyle;
    rightText: IStyle;
    insights: IStyle;
    insightsText: IStyle;
    downloadReport: IStyle;
    chart: IStyle;
    textSection: IStyle;
    radio: IStyle;
    radioOptions: IStyle;
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
            backgroundColor: theme.semanticColors.bodyStandoutBackground, 
            padding: "0 90px",
            height: "90px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between",
            alignItems: "center"
        },
        headerTitle: {
            color: theme.semanticColors.bodyText,
            fontSize: "24px",
            fontWeight: FontWeights.semibold
        },
        headerOptions: {
            backgroundColor: theme.semanticColors.bodyStandoutBackground,
            padding: "0 100px"
        },
        dropDown: {
            margin: "10px 10px 10px 0px",
            display: "inline-block"
        },
        infoButton: {
            color: "#999999",
            float: "left",
            width: "15px",
            height: "15px",
            textAlign: "center",
            fontSize: "12px",
            lineHeight: "14px",
            fontWeight: "600",
            borderRadius: "50%",
            border: "1px solid",
            marginTop: "3px",
            marginRight: "3px"
        },
        doneButton: {
            color: "#FFFFFF",
            borderRadius: "5px",
            background: "#5A53FF",
            padding: "5px 15px",
            selectors: {
                '&:hover': { color: "#ffffff" }
            }
        },
        modalContentIntro: {
            float: 'left',
            textAlign: 'center',
            paddingTop: '10px',
            paddingRight: '20px'
        },
        modalContentHelp: {
            float: 'left',
            textAlign: 'center',
            paddingTop: '10px',
            paddingRight: '20px',
        },
        editButton: {
            color: theme.semanticColors.buttonText,
        },
        howTo: {
            paddingTop: "20px",
            paddingLeft: "100px"
        },
        main: {
            height: "100%",
            flex: 1,
            display: "inline-flex",
            flexDirection: "row",
            backgroundColor: theme.semanticColors.bodyBackground
        },
        mainLeft: {
            width: "75%",
        },
        mainRight: {
            padding: "30px 0 0 35px",
            width: "300px"
        },
        rightTitle: {
            color: theme.semanticColors.bodyText,
            paddingBottom: "18px",
            borderBottom: "1px solid",
            borderColor: theme.semanticColors.bodyDivider
        },
        rightText: {
            padding: "16px 15px 30px 0",
            color: theme.semanticColors.bodyText,
            borderBottom: "0.5px dashed",
            borderColor: theme.semanticColors.bodyDivider
        },
        insights: {
            textTransform: "uppercase",
            color: theme.semanticColors.bodyText,
            padding: "18px 0",
        },
        insightsText: {
            paddingBottom: "18px",
            paddingRight: "15px",
            borderBottom: "1px solid",
            borderColor: theme.semanticColors.bodyDivider
        },
        downloadReport: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            paddingTop: "20px",
            paddingBottom: "20px",
            paddingLeft: "0px",
        },
        chart: {
            padding: "60px 0 0 0",
            flex: 1
        },
        textSection: {
            color: theme.semanticColors.bodyText,
            paddingBottom: "5px"
        },
        radio: {
            paddingBottom: "30px",
            paddingLeft: "75px",
            backgroundColor: theme.semanticColors.bodyBackground
        },
        radioOptions: {
            color: theme.semanticColors.bodyText
        }
    });
};