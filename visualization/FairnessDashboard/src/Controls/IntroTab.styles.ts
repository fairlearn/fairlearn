import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from "office-ui-fabric-react";

export interface IIntroTabStyles {
    firstSection: IStyle;
    firstSectionTitle: IStyle;
    firstSectionSubtitle: IStyle;
    firstSectionBody: IStyle;
    lowerSection: IStyle;
    stepsContainer: IStyle;
    boldStep: IStyle;
    numericLabel: IStyle;
    stepLabel: IStyle;
    explanatoryStep: IStyle;
    explanatoryText: IStyle;
    getStarted: IStyle;
}

export const IntroTabStyles: () => IProcessedStyleSet<IIntroTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IIntroTabStyles>({
        firstSection: {
            padding: "43px 94px",
            //fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`,
            //backgroundColor: "#333333",
            backgroundColor: theme.palette.neutralPrimary,
            color: theme.palette.white
            //color: "#FFFFFF",
        },
        firstSectionTitle: {
            fontSize: "60px",
            //lineHeight: "82px",
            //fontWeight: "100"
            fontWeight: FontWeights.light
        },
        firstSectionSubtitle: {
            fontSize: "60px",
            //lineHeight: "82px",
            //fontWeight: "500"
            fontWeight: FontWeights.semibold
        },
        firstSectionBody: {
            paddingTop: "30px",
            paddingBottom: "70px",
            maxWidth: "500px",
            //color: "#EBEBEB",
            //fontWeight: "300",
            fontWeight: FontWeights.semilight,
            //fontSize: "18px",
            //lineHeight: "24px",
        },
        lowerSection: {
            padding: "50px 70px 90px 90px",
            //backgroundColor: "#F2F2F2",
            //color: "#333333",
            backgroundColor: theme.semanticColors.bodyBackground,
            color: theme.semanticColors.bodyText,
            flexGrow: 1
        },
        stepsContainer: {
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",
            borderBottom: "1px solid #CCCCCC",
            paddingBottom: "38px"
        },
        boldStep: {
            //fontSize: "18px",
            //lineHeight: "24px",
            //fontWeight: "500",
            maxWidth: "300px",
            paddingRight: "25px",
            flex: 1,
        },
        numericLabel: {
            //display:"inline-block",
            //fontSize: "18px",
            fontSize: FontSizes.large,
            //lineHeight: "24px",
            //fontWeight: "700",
            fontWeight: FontWeights.bold,
            color: "#000000",
            width: "30px"
        },
        stepLabel: {
            color: "#333333",
            fontSize: "18px",
            lineHeight: "24px",
            fontWeight: "500",
        },
        explanatoryStep: {
            maxWidth: "300px",
            paddingRight: "20px",
            flex: 1
        },
        explanatoryText: {
            paddingTop: "15px",
            fontSize: "15px",
            lineHeight: "20px",
            color: "#666666"
        },
        getStarted: {
            paddingTop: "30px",
            //color: "#333333",
            //fontSize: "18px",
            fontSize: FontSizes.large,
            //lineHeight: "24px",
            //fontWeight: "500"
            fontWeight: FontWeights.regular
        }
    });
};