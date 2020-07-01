import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from 'office-ui-fabric-react';

export interface IIntroTabStyles {
    firstSection: IStyle;
    firstSectionContainer: IStyle;
    firstSectionTitle: IStyle;
    firstSectionSubtitle: IStyle;
    firstSectionBody: IStyle;
    firstSectionGraphics: IStyle;
    lowerSection: IStyle;
    stepsContainer: IStyle;
    boldStep: IStyle;
    numericLabel: IStyle;
    explanatoryStep: IStyle;
    explanatoryText: IStyle;
    getStarted: IStyle;
}

export const IntroTabStyles: () => IProcessedStyleSet<IIntroTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IIntroTabStyles>({
        firstSection: {
            padding: '43px 94px',
            backgroundColor: theme.semanticColors.bodyBackground,
            color: theme.semanticColors.bodyText,
        },
        firstSectionContainer: {
            width: '100%',
            height: '250px',
        },
        firstSectionTitle: {
            fontSize: '42px',
            lineHeight: '50px',
            fontWeight: FontWeights.light,
        },
        firstSectionSubtitle: {
            fontSize: '42px',
            lineHeight: '50px',
            fontWeight: FontWeights.semibold,
        },
        firstSectionBody: {
            paddingTop: '30px',
            paddingBottom: '70px',
            maxWidth: '500px',
            fontWeight: FontWeights.semilight,
            lineHeight: '24px',
        },
        firstSectionGraphics: {
            width: '346px',
            height: '154px',
            background: theme.semanticColors.bodyBackground,
            fill: theme.semanticColors.bodyText,
            stroke: theme.semanticColors.bodyText,
        },
        lowerSection: {
            padding: '50px 70px 90px 90px',
            backgroundColor: theme.semanticColors.bodyBackground,
            color: theme.semanticColors.bodyText,
            flexGrow: 1,
        },
        stepsContainer: {
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            borderBottom: '1px solid',
            borderBottomColor: theme.semanticColors.bodyDivider,
            paddingBottom: '38px',
        },
        boldStep: {
            maxWidth: '300px',
            paddingRight: '25px',
            flex: 1,
        },
        numericLabel: {
            fontWeight: FontWeights.bold,
            width: '20px',
            marginRight: '5px',
        },
        explanatoryStep: {
            maxWidth: '300px',
            paddingRight: '20px',
            flex: 1,
        },
        explanatoryText: {
            paddingTop: '15px',
        },
        getStarted: {
            height: '44px',
            padding: '12px',
            lineHeight: '24px',
            fontSize: FontSizes.large,
            fontWeight: FontWeights.regular,
        },
    });
};
